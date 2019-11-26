import collections
import enum
import logging
import multiprocessing
import sys
import tqdm
import typing
import uuid


class ResourceStatus(enum.Enum):
    FAILURE = 0
    SUCCESS = 1
    PENDING = 2
    RUNNING = 3
    NOT_READY = 4


CLOSURE = typing.Callable[[], type(None)]

UPQUEUE = multiprocessing.Queue()


def null_function():
    pass


class Resource:

    def __init__(self,
                 fn:CLOSURE,
                 name:str,
                 prepare_fn:CLOSURE = null_function):
        """

        :param fn: The function that must be run to provide the resource
        :param name: The human-readable description of this resource
        :param prepare_fn: a quick-running function to run in the main thread,
        e.g. something to allocate shared memory for the resource.
        """
        self.key = uuid.uuid4()
        self.fn = fn
        self.status = ResourceStatus.PENDING
        self.error = None
        self.name = name
        self.prepare_fn = prepare_fn

    def start_run(self, pool:multiprocessing.Pool):
        logging.info("Starting %s" % self.name)
        try:
            self.prepare_fn()
            del self.prepare_fn
        except:
            logging.exception("Failed running resource %s" % self.key)
            self.report_exception()
            return
        self.status = ResourceStatus.RUNNING
        future = pool.apply_async(Resource.run, (self,))
        return future

    @staticmethod
    def run(self):
        try:
            logging.debug("Running %s" % self.name)
            self.fn()
            logging.debug("Finished %s" % self.name)
            UPQUEUE.put((self.key, ResourceStatus.SUCCESS, None))
        except:
            self.report_exception()

    def report_exception(self):
        logging.exception("Failed running resource %s" % self.key)
        UPQUEUE.put((self.key, ResourceStatus.FAILURE, sys.exc_info()[0]))

    def record_completion(self, status, error):
        logging.info("%s finished with status %s" % (self.name, status))
        self.status = status
        self.error = error
        del self.fn

class Dependent(Resource):
    """A resource which has dependencies that must complete before it can run

    """

    def __init__(self,
                 prerequisites:typing.Sequence[Resource],
                 fn:CLOSURE,
                 name:str,
                 prepare_fn:CLOSURE = null_function):
        """

        :param prerequisites: the resources that must be run before the
        dependent can be run
        :param fn: the function to be run
        """
        super(Dependent, self).__init__(fn, name, prepare_fn=prepare_fn)
        self.prerequisites = dict([(_.key, _) for _ in prerequisites])
        if len(self.prerequisites) > 0:
            self.status = ResourceStatus.NOT_READY

    def requires(self) -> typing.Sequence[Resource]:
        """
        Get the full list of requirements that must be run
        :return: the list of requirements for this dependent- the leaf
        resources
        """
        requirements = []
        for prerequisite in self.prerequisites.values():
            if prerequisite.status == ResourceStatus.RUNNING:
                continue
            if isinstance(prerequisite, Dependent) and \
                    prerequisite.status == ResourceStatus.NOT_READY:
                requirements.extend(prerequisite.requires())
            else:
                requirements.append(prerequisite.key)
        return requirements

    def note_dependency(self,
                        key:uuid.UUID,
                        status:ResourceStatus,
                        error:typing.Any):
        if status == ResourceStatus.FAILURE:
            self.status = status
            self.error = error
        else:
            del self.prerequisites[key]
            if len(self.prerequisites) == 0:
                self.status = ResourceStatus.PENDING


class Pipeline:
    def __init__(self,
                 dependents:typing.Sequence[Dependent]):
        """

        :param resources: the resources that must be run, in the order that
        they should be run
        """
        self.resources = {}
        self.prerequisites = {}
        self.dependents = collections.OrderedDict()
        self.in_flight = set()
        self.progress_bar = None
        self.dependent_keys = [_.key for _ in dependents]
        unevaluated_dependents = [_ for _ in dependents]
        while len(unevaluated_dependents) > 0:
            dependent = unevaluated_dependents.pop(0)
            if dependent.key in self.dependents:
                continue
            self.resources[dependent.key] = dependent
            self.dependents[dependent.key] = dependent
            for prerequisite in dependent.prerequisites.values():
                if prerequisite.key not in self.prerequisites:
                    self.prerequisites[prerequisite.key] = []
                self.prerequisites[prerequisite.key].append(dependent)
                if isinstance(prerequisite, Dependent) and \
                    prerequisite.key not in self.dependents:
                    unevaluated_dependents.append(prerequisite)
                else:
                    self.resources[prerequisite.key] = prerequisite
        self.dependents_per_resource = {}
        for dependent in dependents:
            requirement_keys = tuple(dependent.prerequisites.keys())
            if requirement_keys not in self.dependents_per_resource:
                self.dependents_per_resource[requirement_keys] = []
            self.dependents_per_resource[requirement_keys].append(dependent)

    def run(self, n_workers, silent=False):
        self.progress_bar = tqdm.tqdm(total = len(self.dependents),
                                      disable=silent)
        with multiprocessing.Pool(n_workers) as pool:
            while len(self.dependents) > 0:
                best = None
                best_requirements = None
                best_score = len(self.resources)+1
                to_remove = []
                for resources_key, dependents in \
                        self.dependents_per_resource.items():
                    dependent = dependents[0]
                    if dependent.status == ResourceStatus.NOT_READY:
                        requirements = set(dependent.requires())
                        requirements.difference_update(self.in_flight)
                        if 0 < len(requirements) < best_score:
                            best = dependent
                            best_requirements = requirements
                            best_score = len(requirements)
                    else:
                        to_remove.append(resources_key)
                for key in to_remove:
                    del self.dependents_per_resource[key]
                if best is None:
                    # All dependents' requirements are in-flight
                    # or all dependents are in-flight
                    self.process_upqueue(pool)
                    continue
                for key in best_requirements:
                    requirement = self.resources[key]
                    self.in_flight.add(key)
                    requirement.start_run(pool)
                while len(self.in_flight) > n_workers:
                    self.process_upqueue(pool)
            while len(self.in_flight) > 0:
                self.process_upqueue(pool)
        self.progress_bar.close()

    def process_upqueue(self, pool):
        key, status, error = UPQUEUE.get()
        resource = self.resources[key]
        resource.record_completion(status, error)
        self.in_flight.remove(key)
        if key in self.dependent_keys:
            self.progress_bar.update()
        del self.resources[key]
        if key in self.prerequisites:
            for dependent in self.prerequisites[key]:
                dependent.note_dependency(key, status, error)
                if dependent.status == ResourceStatus.FAILURE:
                    logging.info("Canceling parent of failed: %s" %
                                 dependent.name)
                    del self.dependents[dependent.key]
                elif dependent.status == ResourceStatus.PENDING:
                    dependent.start_run(pool)
                    self.in_flight.add(dependent.key)
                    del self.dependents[dependent.key]
            del self.prerequisites[key]


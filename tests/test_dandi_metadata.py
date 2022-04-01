import contextlib
import io
import sys
import tempfile
import typing
import unittest
from spimstitch.commands.dandi_metadata import main


@contextlib.contextmanager
def make_file(text:str):
    """
    Make a file containing some text, yielding the file name.

    :param text: The text to put into the file
    :return: file name
    """
    with tempfile.NamedTemporaryFile(suffix=".txt") as fd:
        fd.write(text.encode("LATIN-1"))
        fd.flush()
        yield fd.name


def run_dandi_metadata(args:typing.Sequence[str]):
    """
    Run the dandi-metadata command, capturing stdout and returning it

    :param args: the arguments for dandi-metadata
    :return: whatever was spewed by stdout
    """
    fd = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = fd
    try:
        main(args)
        fd.seek(0)
        return fd.read()
    finally:
        sys.stdout = old_stdout


class MyTestCase(unittest.TestCase):
    def test_get_x_step_size(self):
        with make_file(METADATA_WITHOUT_NEGATIVE_Y) as name:
            result = run_dandi_metadata(["get-x-step-size", name])
            self.assertEqual(result.strip(), "1.225")

    def test_get_y_voxel_size(self):
        with make_file(METADATA_WITHOUT_NEGATIVE_Y) as name:
            result = run_dandi_metadata(["get-y-voxel-size", name])
            self.assertEqual(result.strip(), "1.732")

    def test_negative_y(self):
        with make_file(METADATA_WITH_NEGATIVE_Y) as name:
            result = run_dandi_metadata(["get-negative-y", name])
            self.assertEqual(result.strip(), "negative-y")

    def test_not_negative_y(self):
        with make_file(METADATA_WITHOUT_NEGATIVE_Y) as name:
            result = run_dandi_metadata(["get-negative-y", name])
            self.assertEqual(result.strip(), "positive-y")

    def test_not_negative_y_2_1(self):
        with make_file(METADATA_V21) as name:
            result = run_dandi_metadata(["get-negative-y", name])
            self.assertEqual(result.strip(), "negative-y")

    def test_flip_y_legacy(self):
        with make_file(METADATA_WITH_NEGATIVE_Y) as name:
            result = run_dandi_metadata(["get-flip-y", name, "Ex_642_Em_3"])
            self.assertEqual(result.strip(), "flip-y")

    def test_flip_y_488(self):
        with make_file(METADATA_V21) as name:
            result = run_dandi_metadata(["get-flip-y", name, "Ex_488_Em_1"])
            self.assertEqual(result.strip(), "flip-y")

    def test_flip_y_561(self):
        with make_file(METADATA_V21) as name:
            result = run_dandi_metadata(["get-flip-y", name, "Ex_561_Em_2"])
            self.assertEqual(result.strip(), "do-not-flip-y")

    def test_flip_y_642(self):
        with make_file(METADATA_V21) as name:
            result = run_dandi_metadata(["get-flip-y", name, "Ex_642_Em_2"])
            self.assertEqual(result.strip(), "do-not-flip-y")

    def test_flip_y_corners(self):
        for corner_case, expected in (
                ((488 + 561) // 2 - 1, "flip-y"),
                ((488 + 561) // 2 + 1, "do-not-flip-y")):
            with make_file(METADATA_V21) as name:
                result = run_dandi_metadata(["get-flip-y", name,
                                             f"Ex_{corner_case}_Em_2"])
                self.assertEqual(result.strip(), expected)


if __name__ == '__main__':
    unittest.main()


METADATA_WITH_NEGATIVE_Y =\
"""Obj	Res	µm/pix	X/Z Step (µm)	NoOffset02152022	
TL 2x	2048	3.456	2.444		
Power	Left	Right			
488	20	20			
561	20	20			
647	20	20			
X	Y	Z	Laser	Side	Exposure
046560	-136000	-1444	1	0	2
046560	-72300	-1444	1	0	2
046560	-08600	-1444	1	0	2
046560	055100	-1444	1	0	2
046560	-136000	-1444	3	0	2
046560	-136000	-1444	2	0	2
046560	-72300	-1444	3	0	2
046560	-72300	-1444	2	0	2
046560	-08600	-1444	3	0	2
046560	-08600	-1444	2	0	2
046560	055100	-1444	3	0	2
046560	055100	-1444	2	0	2
"""

METADATA_WITHOUT_NEGATIVE_Y=\
"""Obj	Res	µm/pix	X/Z Step (µm)		
LCT 4x	2048	1.732	1.225		
Power	Left	Right			
488	20	20			
561	20	20			
647	20	20			
X	Y	Z	Laser	Side	Exposure
-91830	-07100	2827	1	0	2
-91830	-07100	569	1	0	2
-91830	024820	2827	1	0	2
-91830	024820	569	1	0	2
-91830	056740	2827	1	0	2
-91830	056740	569	1	0	2
-91830	-07100	2827	3	0	2
-91830	-07100	569	3	0	2
-91830	-07100	2827	2	0	2
-91830	-07100	569	2	0	2
-91830	024820	2827	3	0	2
-91830	024820	569	3	0	2
-91830	024820	2827	2	0	2
-91830	024820	569	2	0	2
-91830	056740	2827	3	0	2
-91830	056740	569	3	0	2
-91830	056740	2827	2	0	2
-91830	056740	569	2	0	2
"""

METADATA_V21=\
"""Obj	Res	µm/pix	X/Z Step (µm)	oSPIM3	v2.1
TL 2x	2048	3.226	2.281		NoOffset02152022
Power	Left	Right			
488	20	20			
561	20	20			
X	Y	Z	Laser	Side	Exposure
448910	-53410	3284	1	0	2
448910	-53410	3284	2	0	2
"""
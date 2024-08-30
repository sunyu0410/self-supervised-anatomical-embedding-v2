from pathlib import Path
import pydicom as dicom
import SimpleITK as sitk
import numpy as np


# RTStruct
def get_points(rs_file):
    rs = dicom.read_file(rs_file)
    names = dict([(i.ROINumber, i.ROIName) for i in rs.StructureSetROISequence])
    pts = [
        (i.ReferencedROINumber, i.ContourSequence[0].ContourData)
        for i in rs.ROIContourSequence
        if i.ContourSequence[0].ContourGeometricType == "POINT"
    ]

    return sorted(
        [(num, names[num], coord) for num, coord in pts], key=lambda x: int(x[0])
    )


class Mapper:
    """
    Note that GetOrigin() will return opposite signs for the first two numbers
    https://discourse.slicer.org/t/discrepancy-in-image-origin-values-between-3d-slicer-and-simpleitk/35946/2
    https://slicer.readthedocs.io/en/latest/user_guide/coordinate_systems.html
    Slicer: RAS
    SimpleITK: LPS
    """

    def __init__(self, node):
        self.node = node
        self.matrix = self.getMatrix(self.node)
        self.matrix_inv = np.linalg.inv(self.matrix)

    @staticmethod
    def getMatrix(node):
        ori = node.GetOrigin()
        spa = node.GetSpacing()

        matrix = np.zeros((4, 4), np.float32)
        matrix[0, 0] = spa[0]
        matrix[1, 1] = spa[1]
        matrix[2, 2] = spa[2]
        matrix[3, 3] = 1

        matrix[:3, 3] = ori

        return matrix

    @staticmethod
    def applyMatrix(matrix, coord):
        if len(coord) != 4:
            coord = np.array([*coord, 1])
        return (matrix @ coord)[:3]

    def ijk2lps(self, *coord):
        return self.applyMatrix(self.matrix, coord)

    def lps2ijk(self, *coord):
        return self.applyMatrix(self.matrix_inv, coord).astype(int)


if __name__ == "__main__":
    data_dir = Path(
        r"\\petermac.org.au\shared\ImageStore\Reirradiation\Lung\PMCC_ReIrrad_14"
    )

    reader = sitk.ImageSeriesReader()

    # ids = reader.GetGDCMSeriesIDs(data_dir.as_posix())
    # print(ids)
    # # ('1.2.246.352.221.4988577021572189020.6098343126741388731',
    # #  '1.2.246.352.221.5330343737108144152.7175712638069911460',
    # #  '1.2.246.352.221.5655056896598563174.9427470545387792538')

    # mod = dicom.read_file(reader.GetGDCMSeriesFileNames(data_dir.as_posix(), ids[0])[0]).Modality
    # print(mod)

    fl = reader.GetGDCMSeriesFileNames(
        data_dir.as_posix(), "1.2.246.352.221.5655056896598563174.9427470545387792538"
    )
    reader.SetFileNames(fl)
    image = reader.Execute()
    mapper = Mapper(image)

    pts = get_points(data_dir / "RS.PMCC_ReIrrad_14.CT_LM_062017.dcm")
    pts_ijk = [mapper.lps2ijk(*pt) for (num, label, pt) in pts]
    print(pts)

    # Can now get the ijk for the reference image points

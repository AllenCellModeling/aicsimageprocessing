import numpy as np
import skfmm
from skimage import measure
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import aicsimageio
import time
import mcubes
from aicsimageprocessing import isosurfaceGenerator

# pip install PyMCubes vtk scikit-fmm aicsimageio scikit-image numpy


# notes
# SDF can be generated from a scalar volume + isovalue using skfmm
# SDF can be generated from mesh (mesh generated e.g. as isosurface) 
# jump flooding requires voxels to store location of seed objects: conservative rasterize the isosurface?

# pass in a 3d numpy array of intensities
def make_vtkvolume(npvol):
    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(npvol, npvol.nbytes)
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedShort()
    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of the array it is stored in.
    # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
    # VTK complains if not both are used.
    # TODO: make sure shape[0] is first or last here:
    # dataImporter.SetDataExtent(0, npvol.shape[0]-1, 0, npvol.shape[1]-1, 0, npvol.shape[2]-1)
    # dataImporter.SetWholeExtent(0, npvol.shape[0]-1, 0, npvol.shape[1]-1, 0, npvol.shape[2]-1)
    dataImporter.SetDataExtent(0, npvol.shape[2]-1, 0, npvol.shape[1]-1, 0, npvol.shape[0]-1)
    dataImporter.SetWholeExtent(0, npvol.shape[2]-1, 0, npvol.shape[1]-1, 0, npvol.shape[0]-1)
    dataImporter.Update()
    return dataImporter


def make_vtkpolydata(verts, faces, normals):
    vtkpolydata = vtk.vtkPolyData()

    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(numpy_to_vtk(verts))
    vtkpolydata.SetPoints(vtkpoints)

    # n = vtk.vtkFloatArray()
    # n.SetNumberOfComponents(3)
    # n.SetNumberOfTuples(normals.shape[0])
    # for i in range(normals.shape[0]):
    #     n.SetTuple3(i, normals[i][0], normals[i][1], normals[i][2])
    # n2 = numpy_to_vtk(normals)
    # pointData = vtkpolydata.GetPointData()
    # pointData.SetNormals(n2)

    # Convert to a vtk array
    vtkcells = vtk.vtkCellArray()
    # get number of faces
    nfaces = faces.shape[0]
    idarr = numpy_to_vtkIdTypeArray(faces.ravel())
    vtkcells.SetCells(nfaces, idarr)
    vtkpolydata.SetPolys(vtkcells)
    return vtkpolydata


def generate_sdf_skfmm(im, isovalue, save=False):
    start = time.perf_counter()
    sdf = skfmm.distance(im.astype(np.float32) - isovalue)
    end = time.perf_counter()
    print(f"Generate SDF with skfmm.distance: {end-start} seconds")
    return sdf


def generate_mesh_pymcubes(im, isovalue):
    # generate a mesh using marching cubes:
    start = time.perf_counter()
    vertices, triangles = mcubes.marching_cubes(im, isovalue)
    end = time.perf_counter()
    print(f"Generate mesh with PyMCubes: {end-start} seconds")
    print(f"{len(triangles)} polygons")
    return vertices, triangles


def generate_mesh_vtkmcubes(im, isovalue):
    start = time.perf_counter()
    vtkdataimporter = make_vtkvolume(im)
    # Flying Edges is WAYYYY faster than marching cubes, from vtk.
    # need to compare outputs.  poly count is similar and still 5x the other methods shown.
    vmc = vtk.vtkFlyingEdges3D()
    # vmc = vtk.vtkMarchingCubes()
    vmc.SetInputData(vtkdataimporter.GetOutput())
    vmc.ComputeNormalsOn()
    vmc.ComputeGradientsOff()
    vmc.ComputeScalarsOff()
    vmc.SetValue(0, isovalue)
    vmc.Update()
    ret_vpolydata = vmc.GetOutput()
    end = time.perf_counter()
    print(f"Generate mesh with vtkMarchingCubes: {end-start} seconds")
    print(f"{ret_vpolydata.GetPolys().GetNumberOfCells()} polygons")
    return ret_vpolydata


def generate_mesh_scikitmcubes(im, isovalue):
    start = time.perf_counter()
    verts, faces, normals, values = measure.marching_cubes_lewiner(im, level=isovalue, step_size=1)
    end = time.perf_counter()
    print(f"Generate mesh with marching_cubes_lewiner: {end-start} seconds")
    print(f"{len(faces)} polygons")
    return verts, faces, normals, values


def generate_sdf_vtkmesh(vtkpolydata, im):
    start = time.perf_counter()
    pdd = vtk.vtkImplicitPolyDataDistance()
    pdd.SetInput(vtkpolydata)
    # for each point in 3d volume grid:
    sdf_vtk = np.zeros(im.shape)
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            for k in range(im.shape[2]):
                sdf_vtk[i, j, k] = pdd.EvaluateFunction([i, j, k])
    end = time.perf_counter()
    print(f"Generate SDF with vtkImplicitPolyDataDistance: {end-start} seconds")
    return sdf_vtk


def vtkImg3dToNumpyArray(vtkimageiata):
    x, y, z = vtkimageiata.GetDimensions()
    scalars = vtkimageiata.GetPointData().GetScalars()
    resultingNumpyArray = vtk_to_numpy(scalars)
    resultingNumpyArray = resultingNumpyArray.reshape(z, y, x)
    # transpose?
    return resultingNumpyArray


def generate_sdf2_vtkmesh(vtkpolydata, im):
    start = time.perf_counter()
    pdd = vtk.vtkSignedDistance()
    pdd.SetInputData(vtkpolydata)
    pdd.SetRadius(5.0)
    pdd.SetDimensions(im.shape[2], im.shape[1], im.shape[0])
    pdd.SetBounds(
        0, im.shape[2],
        0, im.shape[1],
        0, im.shape[0],
    )
    pdd.Update()
    vtkimagedata = pdd.GetOutput()
    sdf_vtk = vtkImg3dToNumpyArray(vtkimagedata)
    end = time.perf_counter()
    print(f"Generate SDF with vtkSignedDistance from vtkPolyData: {end-start} seconds")
    return sdf_vtk


def save_sdf(outpath, im):
    start = time.perf_counter()
    wr = aicsimageio.OmeTifWriter(outpath, overwrite_file=True)
    wr.save(im)
    end = time.perf_counter()
    print(f"Save SDF to ome-tiff: {end-start} seconds")


def save_mesh(outpath, points, faces, normals=None):
    start = time.perf_counter()
    m = isosurfaceGenerator.Mesh(points, faces, normals, None)
    m.save_as_obj(outpath)
    end = time.perf_counter()
    print(f"Save mesh to OBJ: {end-start} seconds")


def vtk_iterate_cells(vtkpolydata):
    faces = []
    cells = vtkpolydata.GetPolys()
    idList = vtk.vtkIdList()
    cells.InitTraversal()
    while cells.GetNextCell(idList):
        ids = []
        for i in range(0, idList.GetNumberOfIds()):
            pId = idList.GetId(i)
            ids.append(pId)
        faces.append(ids)
    return faces


def save_mesh_vtk(outpath, vtkpolydata):
    start = time.perf_counter()
    points = vtkpolydata.GetPoints()
    array = points.GetData()
    numpy_points = vtk_to_numpy(array)
    numpy_faces = vtk_iterate_cells(vtkpolydata)
    # cells = vtkpolydata.GetPolys()
    # array = cells.GetData()
    # numpy_faces = vtk_to_numpy(array)
    m = isosurfaceGenerator.Mesh(numpy_points, numpy_faces, None, None)
    m.save_as_obj(outpath)
    end = time.perf_counter()
    print(f"Save vtk mesh to OBJ: {end-start} seconds")


inpath = "D:\\data\\Derek\\3500002980_40X_20190506_C04_LP_2.0_Dwell_4.9us_1_Out-Scene-19.ome.tif"
im = aicsimageio.AICSImage(inpath)
im = im.data[0, 0, :, :, :]
print(f"image loaded and ready: {im.shape[0]}, {im.shape[1]}, {im.shape[2]}")

isovalue = 1500

vpolydata = generate_mesh_vtkmcubes(im, isovalue)
sdf = generate_sdf2_vtkmesh(vpolydata, im)
save_sdf("D:\\data\\sdf\\vtk_sdf.ome.tiff", sdf)

sdf = generate_sdf_skfmm(im, isovalue)
save_sdf("D:\\data\\sdf\\skfmm_sdf.ome.tiff", sdf)

# v, t = generate_mesh_pymcubes(im, isovalue)
# save_mesh("pymcubes.obj", v, t)

# v, t, n, values = generate_mesh_scikitmcubes(im, isovalue)
# save_mesh("scikit.obj", v, t)

# vpolydata = generate_mesh_vtkmcubes(im, isovalue)
# save_mesh_vtk("vtk.obj", vpolydata)
# sdf = generate_sdf_vtkmesh(vpolydata, im)
# save_sdf("vtk_sdf.ome.tiff", sdf)

print("ok")

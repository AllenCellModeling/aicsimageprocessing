import time

import numpy as np
import skfmm
import vtk
from aicsimageio import writers
from skimage import measure
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy

import mcubes
from aicsimageprocessing import isosurfaceGenerator


# pass in a 3d numpy array of intensities
def make_vtkvolume(npvol):
    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be
    # done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    dataImporter.CopyImportVoidPointer(npvol, npvol.nbytes)
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedShort()
    # Because the data that is imported only contains an intensity value (it isnt RGB
    # coded or someting similar), the importer must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of
    # the array it is stored in. I have to admit however, that I honestly dont know the
    # difference between SetDataExtent() and SetWholeExtent() although VTK complains if
    # not both are used.
    dataImporter.SetDataExtent(
        0, npvol.shape[2] - 1, 0, npvol.shape[1] - 1, 0, npvol.shape[0] - 1
    )
    dataImporter.SetWholeExtent(
        0, npvol.shape[2] - 1, 0, npvol.shape[1] - 1, 0, npvol.shape[0] - 1
    )
    dataImporter.Update()
    return dataImporter


def make_vtkpolydata(verts, faces, normals):
    vtkpolydata = vtk.vtkPolyData()

    vtkpoints = vtk.vtkPoints()
    vtkpoints.SetData(numpy_to_vtk(verts))
    vtkpolydata.SetPoints(vtkpoints)

    # TODO insert normals

    # Convert to a vtk array
    vtkcells = vtk.vtkCellArray()
    # get number of faces
    nfaces = faces.shape[0]
    idarr = numpy_to_vtkIdTypeArray(faces.ravel())
    vtkcells.SetCells(nfaces, idarr)
    vtkpolydata.SetPolys(vtkcells)
    return vtkpolydata


def _generate_sdf_skfmm(im, isovalue, save=False):
    start = time.perf_counter()
    sdf = skfmm.distance(im.astype(np.float32) - isovalue)
    end = time.perf_counter()
    print(f"Generate SDF with skfmm.distance: {end-start} seconds")
    return sdf


def _generate_mesh_pymcubes(im, isovalue):
    # generate a mesh using marching cubes:
    start = time.perf_counter()
    vertices, triangles = mcubes.marching_cubes(im, isovalue)
    end = time.perf_counter()
    print(f"Generate mesh with PyMCubes: {end-start} seconds")
    print(f"{len(triangles)} polygons")
    return vertices, triangles


def _generate_mesh_vtkmcubes(im, isovalue):
    start = time.perf_counter()
    vtkdataimporter = make_vtkvolume(im)
    # Flying Edges is WAYYYY faster than marching cubes, from vtk.
    # need to compare outputs. poly count is similar and still 5x the other methods
    # shown.
    vmc = vtk.vtkFlyingEdges3D()
    vmc.SetInputData(vtkdataimporter.GetOutput())
    vmc.ComputeNormalsOn()
    vmc.ComputeGradientsOff()
    vmc.ComputeScalarsOff()
    vmc.SetValue(0, isovalue)
    vmc.Update()
    ret_vpolydata = vmc.GetOutput()
    end = time.perf_counter()
    print(f"Generate mesh with vtkFlyingEdges3D: {end-start} seconds")
    print(f"{ret_vpolydata.GetPolys().GetNumberOfCells()} polygons")
    return ret_vpolydata


def _generate_mesh_scikitmcubes(im, isovalue):
    start = time.perf_counter()
    verts, faces, normals, values = measure.marching_cubes_lewiner(
        im, level=isovalue, step_size=1
    )
    end = time.perf_counter()
    print(
        f"Generate mesh with skimage.measure.marching_cubes_lewiner: {end-start} "
        f"seconds"
    )
    print(f"{len(faces)} polygons")
    return verts, faces, normals, values


def _generate_sdf_vtkmesh(vtkpolydata, im):
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


def _generate_sdf2_vtkmesh(vtkpolydata, im):
    start = time.perf_counter()
    pdd = vtk.vtkSignedDistance()
    pdd.SetInputData(vtkpolydata)
    pdd.SetRadius(0.5 * max(im.shape[2], im.shape[1], im.shape[0]))
    pdd.SetDimensions(im.shape[2], im.shape[1], im.shape[0])
    pdd.SetBounds(
        0, im.shape[2], 0, im.shape[1], 0, im.shape[0],
    )
    pdd.Update()
    vtkimagedata = pdd.GetOutput()
    sdf_vtk = vtkImg3dToNumpyArray(vtkimagedata)
    end = time.perf_counter()
    print(f"Generate SDF with vtkSignedDistance from vtkPolyData: {end-start} seconds")
    return sdf_vtk


def save_sdf(outpath, im):
    start = time.perf_counter()
    wr = writers.ome_tiff_writer.OmeTiffWriter(outpath, overwrite_file=True)
    wr.save(im, dimension_order="ZYX")
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
    # TODO extract normals and maybe uvs too.
    m = isosurfaceGenerator.Mesh(numpy_points, numpy_faces, None, None)
    m.save_as_obj(outpath)
    end = time.perf_counter()
    print(f"Save vtk mesh to OBJ: {end-start} seconds")


def obj_to_sdf(filepath, volume_res):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filepath)
    reader.Update()

    vpolydata = reader.GetOutput()
    bounds = vpolydata.GetBounds()
    print(bounds)

    xform = vtk.vtkTransform()
    xform.PostMultiply()
    xform.Identity()
    xform.Translate(-bounds[0], -bounds[2], -bounds[4])
    xform.Scale(
        volume_res[0] / (bounds[1] - bounds[0]),
        volume_res[1] / (bounds[3] - bounds[2]),
        volume_res[2] / (bounds[5] - bounds[4]),
    )
    xformoperator = vtk.vtkTransformPolyDataFilter()
    xformoperator.SetTransform(xform)
    xformoperator.SetInputConnection(reader.GetOutputPort())
    xformoperator.Update()

    vpolydata = xformoperator.GetOutput()
    bounds = vpolydata.GetBounds()
    print(bounds)

    im = np.zeros(volume_res)
    sdf = _generate_sdf_vtkmesh(vpolydata, im)
    return sdf


# return sdf at same resolution as volume
def volume_to_sdf(im, isovalue=0, method=0):
    if method == 0:
        vpolydata = _generate_mesh_vtkmcubes(im, isovalue)
        sdf = _generate_sdf2_vtkmesh(vpolydata, im)
        return sdf
    elif method == 1:
        sdf = _generate_sdf_skfmm(im, isovalue)
        return sdf
    elif method == 2:
        vpolydata = _generate_mesh_vtkmcubes(im, isovalue)
        sdf = _generate_sdf_vtkmesh(vpolydata, im)
        return sdf


def volume_to_obj(im, isovalue, outpath, method=0):
    if method == 0:
        vpolydata = _generate_mesh_vtkmcubes(im, isovalue)
        save_mesh_vtk(outpath, vpolydata)
    elif method == 1:
        v, t = _generate_mesh_pymcubes(im, isovalue)
        save_mesh(outpath, v, t)
    elif method == 2:
        v, t, n, values = _generate_mesh_scikitmcubes(im, isovalue)
        save_mesh(outpath, v, t)


# take two signed distance fields.
# A has positive values where its mask is 0, and that is considered the "empty"
# available space to fill
# B has positive values where its mask is 0
# result has space between A and B positive
def combine_sdf(A, Amask, B, Bmask):
    start = time.perf_counter()

    Aspace = Amask == 0
    Bspace = Bmask == 0

    inter = Bspace * Aspace

    Aabs = np.absolute(A)
    Babs = np.absolute(B)

    # pick a distance value.  where A and B overlap this could be wrong.
    sub = -np.minimum(Aabs, Babs)
    # invert the spaces that should be positive, in the intersecting "empty" space
    sub = np.where(inter, -sub, sub)

    submask = np.logical_or(Amask > 0, Bmask > 0).astype(float)

    end = time.perf_counter()
    print(f"combine_sdf: {end-start} seconds")

    return sub, submask


# in mask: 1 means filled space resulting in "negative" distances
# 0 means available empty space resulting in "positive" distances
def combine_mask_sdf(Amask, Bmask):
    Cmask = np.logical_or(Amask > 0, Bmask > 0).astype(float)
    C = volume_to_sdf(1 - Cmask, 0.5, 1)
    return C, Cmask

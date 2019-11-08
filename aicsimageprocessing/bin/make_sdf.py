import numpy as np
import skfmm
from skimage import measure, transform
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
import aicsimageio
from aicsimageio import writers, AICSImage
import time
import mcubes
# import pywavefront
from aicsimageprocessing import isosurfaceGenerator, textureAtlas

# pip install PyMCubes vtk scikit-fmm aicsimageio scikit-image numpy pywavefront


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
    print(f"Generate mesh with vtkFlyingEdges3D: {end-start} seconds")
    print(f"{ret_vpolydata.GetPolys().GetNumberOfCells()} polygons")
    return ret_vpolydata


def _generate_mesh_scikitmcubes(im, isovalue):
    start = time.perf_counter()
    verts, faces, normals, values = measure.marching_cubes_lewiner(im, level=isovalue, step_size=1)
    end = time.perf_counter()
    print(f"Generate mesh with skimage.measure.marching_cubes_lewiner: {end-start} seconds")
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
    pdd.SetRadius(0.5*max(im.shape[2], im.shape[1], im.shape[0]))
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
    # cells = vtkpolydata.GetPolys()
    # array = cells.GetData()
    # numpy_faces = vtk_to_numpy(array)
    m = isosurfaceGenerator.Mesh(numpy_points, numpy_faces, None, None)
    m.save_as_obj(outpath)
    end = time.perf_counter()
    print(f"Save vtk mesh to OBJ: {end-start} seconds")


def obj_to_sdf(filepath, volume_res):
    # scene = pywavefront.Wavefront(filepath)
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
    xform.Scale(volume_res[0]/(bounds[1]-bounds[0]), volume_res[1]/(bounds[3]-bounds[2]), volume_res[2]/(bounds[5]-bounds[4]))
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
        # v, t = generate_mesh_pymcubes(im, isovalue)
        # v, t, n, values = generate_mesh_scikitmcubes(im, isovalue)
        vpolydata = _generate_mesh_vtkmcubes(im, isovalue)
        # save_mesh_vtk("vtk.obj", vpolydata)
        sdf = _generate_sdf_vtkmesh(vpolydata, im)
        return sdf
        # save_sdf("vtk_sdf.ome.tiff", sdf)


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
# A has positive values where its mask is 0, and that is considered the "empty" available space to fill
# B has positive values where its mask is 0
# result has space between A and B positive
def combine_sdf(A, Amask, B, Bmask):
    start = time.perf_counter()

    Aspace = (Amask == 0)
    Bspace = (Bmask == 0)

    inter = Bspace * Aspace

    Aabs = np.absolute(A)
    Babs = np.absolute(B)

    # pick a distance value.  where A and B overlap this could be wrong.
    sub = -np.minimum(Aabs, Babs)
    # invert the spaces that should be positive, in the intersecting "empty" space
    sub = np.where(inter, -sub, sub)

    # Afilled = (Amask == 1)
    # Bfilled = (Bmask == 1)
    # interfilled = Bfilled * Afilled
    # submax = -np.maximum(Aabs, Babs)

    # sub = np.where(interfilled, -submax, sub)

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


def main():
    out_dir = "E:\\data\\sdf\\test_meshes\\"
    obj_to_test = "E:\\data\\sdf\\test_meshes\\InsideOutside_SingleTestMesh_1_coarse-1.obj"

    phys_size = (0.29, 0.1083, 0.1083)
    phys_scaling = tuple(i / min(phys_size) for i in phys_size)
    raw = "E:\\data\\sdf\\3500000968_100X_20170612_2-Scene-22-P51-F07.ome.tiff"
    segs = [
        ("dna", "E:\\data\\sdf\\3500000968_100X_20170612_2-Scene-22-P51-F07.ome_dna_segmentation.tiff"),
        ("mem", "E:\\data\\sdf\\3500000968_100X_20170612_2-Scene-22-P51-F07.ome_mem_segmentation.tiff"),
        ("struct", "E:\\data\\sdf\\3500000968_100X_20170612_2-Scene-22-P51-F07_struct_segmentation.tiff"),
    ]
    with aicsimageio.AICSImage(segs[0][1]) as im:
        dna_vol = im.get_image_data(out_orientation="ZYX", C=0, T=0, S=0)
        dna_vol = dna_vol.astype(np.float32)
    with aicsimageio.AICSImage(segs[1][1]) as im:
        mem_vol = im.get_image_data(out_orientation="ZYX", C=0, T=0, S=0)
        mem_vol = mem_vol.astype(np.float32)
    with aicsimageio.AICSImage(segs[2][1]) as im:
        struct_vol = im.get_image_data(out_orientation="ZYX", C=0, T=0, S=0)
        struct_vol = struct_vol.astype(np.float32)

    with aicsimageio.AICSImage(raw) as im:
        raw_mem = im.get_image_data(out_orientation="ZYX", C=1, T=0, S=0)
        raw_mem = raw_mem.astype(np.float32)
        raw_struct = im.get_image_data(out_orientation="ZYX", C=3, T=0, S=0)
        raw_struct = raw_struct.astype(np.float32)
        raw_dna = im.get_image_data(out_orientation="ZYX", C=5, T=0, S=0)
        raw_dna = raw_dna.astype(np.float32)

    unique_elements = np.unique(mem_vol)
    print(unique_elements)
    # unique_elements = np.unique(dna_vol)
    # print(unique_elements)

    for k in unique_elements:
        if k == 0:
            continue
        tmp = np.copy(mem_vol)
        tmp[tmp != k] = 0
        coords = np.argwhere(tmp > 0)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        cropped_cell = tmp[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

        cropped_dna = dna_vol[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        cropped_dna[cropped_dna != k] = 0
        cropped_struct = struct_vol[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        cropped_struct[cropped_cell != k] = 0

        cropped_cell = transform.rescale(cropped_cell, phys_scaling, multichannel=False)
        cropped_cell[cropped_cell != 0] = 1
        cropped_dna = transform.rescale(cropped_dna, phys_scaling, multichannel=False)
        cropped_dna[cropped_dna != 0] = 1
        cropped_struct = transform.rescale(cropped_struct, phys_scaling, multichannel=False)
        cropped_struct[cropped_struct != 0] = 1

        cropped_raw_mem = raw_mem[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        cropped_raw_dna = raw_dna[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        cropped_raw_struct = raw_struct[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        cropped_raw_mem = transform.rescale(cropped_raw_mem, phys_scaling, multichannel=False)
        cropped_raw_dna = transform.rescale(cropped_raw_dna, phys_scaling, multichannel=False)
        cropped_raw_struct = transform.rescale(cropped_raw_struct, phys_scaling, multichannel=False)

        # three = np.array([cropped_cell, cropped_dna, cropped_struct])
        # wr = writers.ome_tiff_writer.OmeTiffWriter(f"{out_dir}cell3_{k}.ome.tiff", overwrite_file=True)
        # wr.save(three, dimension_order="CZYX")

        # make a SDF for each of the three
        sdf_cell = volume_to_sdf(cropped_cell, 0.5, 1)
        sdf_dna = volume_to_sdf(cropped_dna, 0.5, 1)
        sdf_struct = volume_to_sdf(cropped_struct, 0.5, 1)

        # invert the distance values
        # three = np.array([-sdf_cell, -sdf_dna, -sdf_struct])
        # wr = writers.ome_tiff_writer.OmeTiffWriter(f"{out_dir}sdf3_{k}.ome.tiff", overwrite_file=True)
        # wr.save(three, dimension_order="CZYX")

        # now compute combined fields.

        CD, CDmask = combine_sdf(-sdf_cell, 1 - cropped_cell, sdf_dna, cropped_dna)
        CDM, CDMmask = combine_sdf(CD, CDmask, sdf_struct, cropped_struct)

        ##############################
        CD2, CDmask2 = combine_mask_sdf(1 - cropped_cell, cropped_dna)
        CDM2, CDMmask2 = combine_mask_sdf(CDmask2, cropped_struct)

        # TODO make a uint16 version of this
        image = np.array([CD, CDM, CD2, CDM2, CDmask, CDMmask,
                         cropped_raw_mem, cropped_raw_dna, cropped_raw_struct])
        mins = [image[i].min() for i in range(image.shape[0])]
        maxs = [image[i].max() for i in range(image.shape[0])]
        # these isos only make sense for the SDFs
        isos = [-mins[i]/(maxs[i]-mins[i]) for i in range(len(mins))]
        # TODO: squish back to original sizeZ ?
        channel_names = ["C-D approx", "C-D-M approx", "C-D from mask", "C-D-M from mask", "C-D mask", "C-D-M mask",
                         "raw mem", "raw dna", "raw mito"]
        isovalues = [
                        isos[0],
                        isos[1],
                        isos[2],
                        isos[3],
                        0.5,
                        0.5,
                        0, 0, 0
                    ]

        a = textureAtlas.generate_texture_atlas(AICSImage(image), name=f"SDFtest_{k}", max_edge=2048, pack_order=None)
        a.save(out_dir, name=f"SDFtest_{k}",
               user_data={"isovalues": isovalues})

        # create a uint16 version of image
        def to16(a):
            m = a.min()
            if m < 0:
                a = a + a.min()
            a = a / a.max()
            a = a * 65535.0
            return a.astype(np.uint16)
        image = np.array([to16(CD),
                         to16(CDM),
                         to16(CD2),
                         to16(CDM2),
                         CDmask.astype(np.uint16),
                         CDMmask.astype(np.uint16),
                         cropped_raw_mem.astype(np.uint16),
                         cropped_raw_dna.astype(np.uint16),
                         cropped_raw_struct.astype(np.uint16)])

        wr = writers.ome_tiff_writer.OmeTiffWriter(f"{out_dir}SDFtest_{k}.ome.tiff", overwrite_file=True)
        wr.save(image,
                dimension_order="CZYX",
                channel_names=channel_names)

        print("ok")
        break

    # sdfs = []
    # for name, segfile in segs:
    #     with aicsimageio.AICSImage(segfile) as im:
    #         vol = im.get_image_data(out_orientation="ZYX", C=0, T=0, S=0)
    #         vol = vol.astype(np.float32)
    #         # downsample the volume to something "small enough"
    #         vol = transform.rescale(vol, phys_scaling, anti_aliasing=True, multichannel=False)
    #         # method 1 does not require the surface to be hollow
    #         sdf = volume_to_sdf(vol, 0.5, 1)
    #         sdfs.append(sdf)
    #         save_sdf(f"{out_dir}sdf_{name}.ome.tiff", sdf)

    # print("===Mesh to SDF at 128^3")
    # sdf = obj_to_sdf(obj_to_test, (128, 128, 128))
    # save_sdf(out_dir + "sdf_frommesh_128.ome.tiff", sdf)

    # print("===Mesh to SDF at 256^3")
    # sdf = obj_to_sdf(obj_to_test, (256, 256, 256))
    # save_sdf(out_dir + "sdf_frommesh_256.ome.tiff", sdf)

    inpath = "E:\\data\\Derek\\3500002980_40X_20190506_C04_LP_2.0_Dwell_4.9us_1_Out-Scene-19.ome.tif"

    # get the first channel of the volume image.
    with aicsimageio.AICSImage(inpath) as im:
        vol = im.get_image_data(out_orientation="ZYX", C=0, T=0, S=0)
    print(f"image loaded and ready: {vol.shape[0]}, {vol.shape[1]}, {vol.shape[2]}")

    isovalue = 1500

    # print("===Volume to SDF method 0")
    # sdf = volume_to_sdf(vol, isovalue, 0)
    # save_sdf(out_dir + "sdf_fromvol_1500_0.ome.tiff", sdf)

    print("===Volume to SDF method 1")
    sdf = volume_to_sdf(vol, isovalue, 1)
    save_sdf(out_dir + "sdf_fromvol_1500_1.ome.tiff", sdf)

    # print("===Volume to SDF method 2")
    # sdf = volume_to_sdf(vol, isovalue, 2)
    # save_sdf(out_dir + "sdf_fromvol_1500_2.ome.tiff", sdf)

    print("===Volume to mesh method 0")
    volume_to_obj(vol, isovalue, out_dir + "iso_1500_0.obj", 0)

    print("===Volume to mesh method 1")
    volume_to_obj(vol, isovalue, out_dir + "iso_1500_1.obj", 1)

    print("===Volume to mesh method 2")
    volume_to_obj(vol, isovalue, out_dir + "iso_1500_2.obj", 2)


if __name__ == "__main__":
    main()

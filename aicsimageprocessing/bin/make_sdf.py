import numpy as np
from skimage import transform
import aicsimageio
from aicsimageio import writers, AICSImage
# import pywavefront
from aicsimageprocessing import sdfGenerator, textureAtlas

# pip install PyMCubes vtk scikit-fmm aicsimageio scikit-image numpy pywavefront


# notes
# SDF can be generated from a scalar volume + isovalue using skfmm
# SDF can be generated from mesh (mesh generated e.g. as isosurface)
# jump flooding requires voxels to store location of seed objects: conservative rasterize the isosurface?


def main():
    out_dir = "E:\\data\\sdf\\test_meshes\\"
    # obj_to_test = "E:\\data\\sdf\\test_meshes\\InsideOutside_SingleTestMesh_1_coarse-1.obj"

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
        sdf_cell = sdfGenerator.volume_to_sdf(cropped_cell, 0.5, 1)
        sdf_dna = sdfGenerator.volume_to_sdf(cropped_dna, 0.5, 1)
        sdf_struct = sdfGenerator.volume_to_sdf(cropped_struct, 0.5, 1)

        # invert the distance values
        # three = np.array([-sdf_cell, -sdf_dna, -sdf_struct])
        # wr = writers.ome_tiff_writer.OmeTiffWriter(f"{out_dir}sdf3_{k}.ome.tiff", overwrite_file=True)
        # wr.save(three, dimension_order="CZYX")

        # now compute combined fields.

        CD, CDmask = sdfGenerator.combine_sdf(-sdf_cell, 1 - cropped_cell, sdf_dna, cropped_dna)
        CDM, CDMmask = sdfGenerator.combine_sdf(CD, CDmask, sdf_struct, cropped_struct)

        ##############################
        CD2, CDmask2 = sdfGenerator.combine_mask_sdf(1 - cropped_cell, cropped_dna)
        CDM2, CDMmask2 = sdfGenerator.combine_mask_sdf(CDmask2, cropped_struct)

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
    sdf = sdfGenerator.volume_to_sdf(vol, isovalue, 1)
    sdfGenerator.save_sdf(out_dir + "sdf_fromvol_1500_1.ome.tiff", sdf)

    # print("===Volume to SDF method 2")
    # sdf = volume_to_sdf(vol, isovalue, 2)
    # save_sdf(out_dir + "sdf_fromvol_1500_2.ome.tiff", sdf)

    print("===Volume to mesh method 0")
    sdfGenerator.volume_to_obj(vol, isovalue, out_dir + "iso_1500_0.obj", 0)

    print("===Volume to mesh method 1")
    sdfGenerator.volume_to_obj(vol, isovalue, out_dir + "iso_1500_1.obj", 1)

    print("===Volume to mesh method 2")
    sdfGenerator.volume_to_obj(vol, isovalue, out_dir + "iso_1500_2.obj", 2)


if __name__ == "__main__":
    main()

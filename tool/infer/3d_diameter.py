# 用mesh进行3d重建，之后计算管径
import argparse
import os

import skimage.measure
import scipy.ndimage
import numpy as np
import nibabel as nib
import trimesh
from util import filter_polygon, sort_line, Polygon

parser = argparse.ArgumentParser()
parser.add_argument(
    "--in_dir", type=str, default="/home/lin/Desktop/aorta/private/label/test"
)
parser.add_argument("--out_dir", type=str, default="./img")
args = parser.parse_args()


vol_names = os.listdir(args.in_dir)
for vol_name in vol_names:
    # 1. 获取需要测量的标签数据，插值成1024分辨率提升精度
    volf = nib.load(os.path.join(args.in_dir, vol_name))
    vol = volf.get_fdata()
    print(vol.shape)
    # if vol.shape[0] < 1024:
    #     vol = scipy.ndimage.interpolation.zoom(vol, (2, 2, 1), order=3)
    # 2. 进行3D重建
    verts, faces, normals, values = skimage.measure.marching_cubes(vol)
    # print(verts)
    verts = [[v[0], v[1], v[2] * 10] for v in verts]
    # print(verts)
    # 将bb左下角放到原点
    min = np.min(verts, axis=0)
    print(min)
    verts = verts - min
    # print(verts)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    # print(mesh.is_watertight)
    # print(mesh.bounds)

    # mesh.export("aorta.stl")
    # mesh = trimesh.load("./aorta.stl")

    scene = trimesh.Scene()
    scene.add_geometry(mesh)
    axis = trimesh.creation.axis(origin_size=10, origin_color=[1.0, 0, 0])
    scene.add_geometry(axis)
    scene.show()

    # 3. 获取测量路径，在血管壁上取一条线的点
    # 3.1 查所有不同的高度，作为一个片曾
    heights = []
    for v in verts:
        if v[2] not in heights:
            heights.append(v[2])
    print("heights", heights)

    slices = [[] for _ in range(len(heights))]
    for ind, h in enumerate(heights):
        for v in verts:
            if v[2] == h:
                slices[ind].append(v)

    # 3.2 算所有片曾的圆心
    centers = []
    polygons = []
    for ind in range(len(heights)):
        res = filter_polygon(slices[ind], "all", 15)
        for poly in res:
            polygons.append(Polygon(poly))

    polygons = sort_line(polygons)

    # center_cloud = trimesh.points.PointCloud([a.center for a in polygons], [0, 255, 0, 100])
    # scene.add_geometry(center_cloud)
    #
    # base_cloud = trimesh.points.PointCloud([a.base for a in polygons], [255, 0, 0, 100])
    # scene.add_geometry(base_cloud)
    # scene.show()

    # 4. 找每个一base和血管相交，最小的圆
    for polygon in polygons[10:]:
        base = polygon.base
        last_prin = [0, 0, 1]
        last_size = 65535
        diameters = []
        while True:
            stride = 0.01
            tweak = [
                [0, 0, stride],
                [0, 0, -stride],
                [0, stride, 0],
                [0, -stride, 0],
                [stride, 0, 0],
                [-stride, 0, 0],
            ]
            new_prins = np.array(last_prin) + np.array(tweak)
            print(new_prins)
            sizes = []

            for prin in new_prins:
                print(prin)
                lines = trimesh.intersections.mesh_plane(mesh, prin, base)
                size = Polygon(lines).cal_size()
                if size < 5:
                    size = 65535
                sizes.append(size)
            print(sizes)

            min_ind = np.array(sizes).argmin()
            if last_size > sizes[min_ind]:
                last_prin = new_prins[min_ind]
                last_size = sizes[min_ind]
                print("+_+_+_+_", last_size)
            else:
                break

        lines = trimesh.intersections.mesh_plane(mesh, last_prin, base)
        min_polygon = Polygon(lines)
        diameters.append(min_polygon.cal_diameter())
        center_cloud = trimesh.points.PointCloud(polygon.points, [0, 255, 0, 100])
        scene.add_geometry(center_cloud)
        center_cloud = trimesh.points.PointCloud(min_polygon.points, [255, 0, 0, 100])
        scene.add_geometry(center_cloud)
        scene.show()
        print("get_min")
        input("here")

    # points = []
    # for p in lines:
    #     points.append([p[0][0], p[0][1], p[0][2]])
    # points = ang_sort(points)
    # print(points)
    # points = filter_polygon(points, [0, 0, 0], 10)

    # points = trimesh.points.PointCloud(points, [200, 200, 250, 100])

    # faces = []
    # for ind in range(len(points)):
    # faces.append([ind, int((ind + len(points) / 2) % len(points)), int((ind + len(points) / 2 + 1) % len(points))])
    # plane = trimesh.creation.extrude_triangulation(points, [[ind, ind + 1, ind + 2] for ind in range(20)], height=-1)


import numpy as np
import pyvista as pv
import pyvistaqt as pvqt

from numpy.typing import ArrayLike
import time


def addCamera(plotter: pv.Plotter, cam_name: str, cam_t: ArrayLike, cam_R: ArrayLike, scale=0.04):
    points = np.array([[1.0,  0.8, 3.0],
                       [-1.0,  0.8, 3.0],
                       [-1.0, -0.8, 3.0],
                       [1.0, -0.8, 3.0],
                       [0.0, 0.0, 0.0]]) * 0.5 * scale
    points = (points - np.asarray(cam_t).reshape(1, 3)) @ np.asarray(cam_R).reshape(3, 3)
    cam_mesh = pv.Pyramid(points)

    plotter.add_mesh(cam_mesh)
    plotter.add_point_labels(points[-1:], [cam_name], shape_opacity=0.2)


def addPoints(plotter: pv.Plotter, points: ArrayLike, size:float = 0.1):
    # points: the first frame of the points,
    #         the shape should be either [N, 3] or [N, m, 3],
    #         where m is the number of points of a face
    points = np.asarray(points)
    assert points.ndim == 2 or points.ndim == 3
    assert points.shape[-1] == 3

    points = points.copy()
    points[abs(points) > 10] = 0

    if points.ndim == 2:
        mesh = plotter.add_points(points, render_points_as_spheres=True, point_size=size)
        return mesh

    faces = [([points.shape[1]] + [i*points.shape[1] + j for j in range(points.shape[1])]) for i in range(points.shape[0])]
    mesh = pv.PolyData(points.reshape(-1, 3), faces=faces)
    plotter.add_mesh(mesh)
    return mesh


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-c', '--camera-ids', type=str, default='1,2,3,4,5,6,7,8,9,10,11,12,13')
    parser.add_argument('-p', '--points', type=str, default="./Out/synth_01/triangulation/hand_patches_markers_3d_positions.npy")

    args = parser.parse_args()

    points = None
    if args.points is not None:
        plotter = pvqt.BackgroundPlotter(show=True, auto_update=60)

        points = np.load(args.points)
        points = points[0]
        mesh = addPoints(plotter, points, 4)
    else:
        plotter = pv.Plotter()

    # pv.set_plot_theme("paraview")
    # pv.set_plot_theme("dark")

    camera_ids = np.array(list(map(int, args.camera_ids.strip('()"\',').split(','))))
    print(f'> Camera IDs: {camera_ids}')

    camera_names = os.listdir(args.input)

    for cam_id, cam_name in enumerate(camera_names):
        if not cam_id+1 in camera_ids:
            continue

        with np.load(os.path.join(args.input, cam_name, f'{cam_name}_extrinsics.npz')) as f:
            addCamera(plotter, cam_name, f["tvecs"], f["rmats"])

    plotter.add_mesh(pv.Arrow(direction=(1, 0, 0), scale=0.1), color='r')
    plotter.add_mesh(pv.Arrow(direction=(0, 1, 0), scale=0.1), color='g')
    plotter.add_mesh(pv.Arrow(direction=(0, 0, 1), scale=0.1), color='b')
    plotter.show_axes()
    # plotter.view_xz()
    plotter.view_isometric()
    plotter.show_bounds(grid='front', location='outer', all_edges=False, bold=False)

    if points is None:
        plotter.show()
        return

    plotter.show()

    def update_points():
        cnt = 1
        while plotter.app_window.isVisible():
            mesh.points = points[cnt].reshape(-1, 3)
            cnt = (cnt + 1) % points.shape[0]
            time.sleep(1/60)

    from threading import Thread
    thread = Thread()
    thread.start()

    plotter.app.exec()


if __name__ == '__main__':
    main()

import argparse
import os
from typing import List

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from VideoProcess.utils import load_label_patches, get_edges_from_block_def


def init_rerun(name: str, camera_ids: List[int]):
    """
    Initialize the rerun visualization.

    Parameters:
    - name (str): The name of the rerun session.
    """
    # rr.ViewCoordinates.
    rr.init(name, spawn=True)
    rr.log("", rr.ViewCoordinates.RIGHT_HAND_Z_DOWN, static=True)  # Set an up-axis = +Y
    rr.set_time("stable_time", duration=0)

    # # visualize axes
    # rr.log(
    #     "arrows",
    #     rr.Arrows3D(
    #         vectors=[[1,0,0], [0,1,0], [0,0,1]],
    #         radii=0.003,
    #         labels=['x', 'y', 'z']
    #     )
    # )

    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(name="3D"),
        rrb.Vertical(
            rrb.Tabs(
                # Note that we re-project the annotations into the 2D views:
                # For this to work, the origin of the 2D views has to be a pinhole camera,
                # this way the viewer knows how to project the 3D annotations into the 2D views.
                *(
                    rrb.Spatial2DView(
                        name=f"cam_{cam_id}",
                        origin=f"camera/camera_{cam_id}",
                        contents=[
                            f"$origin/**",
                            # f"marker/**",
                        ],
                    )
                    for cam_id in camera_ids
                ),
                name="2D",
            ),
            # rrb.TextDocumentView(name="Readme"),
            # row_shares=[2, 1],
        ),
    )

    rr.send_blueprint(blueprint)


def visualize_cameras(camera_ids: List[int], folder):
    camera_names = os.listdir(folder)

    for cam_id in camera_ids:

        with np.load(os.path.join(folder, camera_names[cam_id-1], f'{camera_names[cam_id-1]}_extrinsics.npz')) as f:
            instrinsic = f["intrinsics"]
            tvec, rmat = f["tvecs"], f["rmats"]

        # rr.log(f"camera/label_{cam_id}",
        #     rr.Points3D(
        #         positions=[0,0,0],
        #         radii=0.01,
        #         labels=cam_id
        #     )
        # )
        # rr.log(
        #     f"camera/label_{cam_id}",
        #     rr.Transform3D(mat3x3=rmat, translation=tvec, from_parent=True),
        # )

        rr.log(
            f"camera/camera_{cam_id}",
            rr.Pinhole(
                image_from_camera=instrinsic,
                width=2448,
                height=2048,
                # camera_xyz=rr.ViewCoordinates.RDF
            ),
        )

        rr.log(
            f"camera/camera_{cam_id}",
            rr.Transform3D(mat3x3=rmat, translation=tvec, from_parent=True),
        )
        # rr.log(f"camera/camera_{cam_id}", rr.Pinhole(image_from_camera=instrinsic, width=2448, height=2048))
        # rr.log(f"camera/camera_{cam_id}", rr.Transform3D(mat3x3=rmat, translation=tvec, relation=rr.TransformRelation.ChildFromParent))



def visualize_2d_points(
    camera_ids: List[int],
    points_2d: np.ndarray,
    edges: List[List[int]],
    color: np.ndarray | List = [60, 200, 205],
    edge_color: np.ndarray | List = [128, 128, 128],
):
    edge_indices = np.asarray(edges).reshape(-1, 2)

    num_cam, num_points = points_2d.shape[:2]
    for cam_idx, cam_id in enumerate(camera_ids):
        rr.log(
            f"camera/camera_{cam_id}/image_point_{cam_id}",
            rr.Points2D(
                positions=points_2d[cam_idx],
                radii=5,
                colors=color,
                # labels=cam_id
            ),
        )

        edge_points = points_2d[cam_idx, edge_indices].reshape(-1, 2, 2)
        edge_points[np.any(edge_points <= 0, axis=(1, 2))] = 0

        rr.log(
            f"camera/camera_{cam_id}/image_edges_{cam_id}",
            rr.LineStrips2D(
                edge_points,
                radii=1,
                colors=edge_color,
                # labels=cam_id
            ),
        )


def visualize_points(points: np.ndarray, color: np.ndarray | List = [255, 255, 60]):
    npts, ndim = points.shape
    for j in range(npts):
        rr.log(
            f"marker/marker_{j}",
            rr.Points3D(
                positions=points[j],
                radii=0.001,
                colors=color,
                # labels=body_part_indices[j]
            ),
        )


def visualize_edges(
    points: np.ndarray,
    edges: List[List[int]],
    color: np.ndarray | List = [128, 128, 128],
):
    edge_points = points[np.asarray(edges).reshape(-1, 2)].reshape(-1, 2, 3)
    edge_points[np.any(edge_points == 0, axis=(1, 2))] = 0

    rr.log("edges", rr.LineStrips3D(edge_points, radii=0.0002, colors=color))


def visualize_blocks(points, blocks):
    for label in blocks:
        marker_indices = blocks[label]["markers"]
        exist = True
        for idx in marker_indices:
            if np.any(points[idx] == 0):
                exist = False
                break

        if exist:
            center = np.mean(points[marker_indices, :], axis=0)
        else:
            center = np.zeros(3)

        rr.log(
            f"block/block_{label}",
            rr.Points3D(positions=center, radii=0.00001, labels=label),
        )


def visualize():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help='Camera subfolders each containing _extrinsics.npz files')

    parser.add_argument("-f", "--file", type=str, required=True, help='Either _triangulation.npz file or _markers_3d_positions.npy file')
    parser.add_argument("--camera-ids", default="1,2,3,4,5,6,7,8,9,10,11,12,13", type=str, help="id of cameras, in the fmt of 2,3,4...")
    parser.add_argument('-l', "--label-patches", type=str, required=True, help="label_patches.json")

    parser.add_argument("--fps", type=int, default=20)

    args = parser.parse_args()

    camera_ids = np.array(list(map(int, args.camera_ids.strip("()\"',").split(","))))
    print(f"> Camera_ids: {camera_ids}")
    fps = args.fps

    marker_defs, block_defs, patches = load_label_patches(args.label_patches)
    edges = get_edges_from_block_def(block_defs=block_defs)

    data = np.load(args.file)

    use_visualize_2d = False

    if 'markers_undistorted' in data:
        use_visualize_2d = True
        points_3d = data["markers_3d_positions"]  # [nframe, npts, 3]
        points_2d = data["markers_undistorted"]  # [nframe, ncam, npts, 3]
    else:
        points_3d = data

    points_3d[np.all(points_3d == [-1e3, -1e3, -1e3], axis=-1)] = 0

    if use_visualize_2d:
        invalid_points = points_2d[..., 0] < 0
        points_2d[invalid_points] = -1

    if use_visualize_2d:
        num_frames, num_cameras, num_points = points_2d.shape[:3]
        assert num_cameras == len(camera_ids)
    else:
        num_frames = points_3d.shape[0]

    init_rerun("Triangulation Visualization (2D & 3D)" if use_visualize_2d else "Triangulation Visualization (3D only)", camera_ids=camera_ids)
    visualize_cameras(camera_ids, args.input)

    for frame in range(num_frames):
        rr.set_time("stable_time", duration=frame / fps)

        visualize_points(points_3d[frame])
        if use_visualize_2d:
            visualize_2d_points(camera_ids, points_2d[frame], edges)
        visualize_edges(points_3d[frame], edges)
        visualize_blocks(points_3d[frame], block_defs)

    input("Press any key to close")


if __name__ == "__main__":
    visualize()

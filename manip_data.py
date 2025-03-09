"""
Manipulation of result data including cropping, remake of video with upscaling and etc.
"""

from FreeTrace.module.image_module import make_loc_radius_video, remake_visual_trajectories, remake_visual_localizations, make_loc_depth_image
from FreeTrace.module.auxiliary import crop_trace_roi_and_frame


if __name__ == '__main__':
    """
    output_path = 'outputs'
    coords_path = ['outputs/Tubulin-A647-3D-stacks_1_0_loc.csv', 'outputs/Tubulin-A647-3D-stacks_1_1_loc.csv', 'outputs/Tubulin-A647-3D-stacks_2_0_loc.csv', 'outputs/Tubulin-A647-3D-stacks_2_1_loc.csv',
                   'outputs/Tubulin-A647-3D-stacks_3_0_loc.csv', 'outputs/Tubulin-A647-3D-stacks_3_1_loc.csv', 'outputs/Tubulin-A647-3D-stacks_4_0_loc.csv', 'outputs/Tubulin-A647-3D-stacks_4_1_loc.csv']
    make_loc_depth_image(output_path, coords_path, multiplier=4, winsize=7, resolution=2, dim=3)
    """

    trace_path = 'outputs/sample0_traces.csv'
    roi_path = ''
    crop_trace_roi_and_frame(trace_path, roi_path, start_frame=1, end_frame=50)


    output_path = 'outputs'
    images = 'inputs/sample0.tiff'
    localization_file = 'outputs/sample0_loc.csv'
    make_loc_radius_video(output_path, images, localization_file, frame_cumul=10, radius=[3, 25], start_frame=0, end_frame=100, alpha1=0.65, alpha2=0.35)
    

    output_path = 'outputs'
    images = 'inputs/sample0.tiff'
    localization_file = 'outputs/sample0_loc.csv'
    trajectory_file = 'outputs/sample0_traces.csv'
    remake_visual_localizations(output_path, localization_file, images, start_frame=1, end_frame=100, upscaling=1)
    remake_visual_trajectories(output_path, trajectory_file, images, start_frame=1, end_frame=100, upscaling=1)

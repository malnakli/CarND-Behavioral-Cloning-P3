from moviepy.editor import ImageSequenceClip
import argparse


def main():
    parser = argparse.ArgumentParser(description='Create driving video.')
    parser.add_argument(
        'image_folder',
        type=str,
        default='',
        help='Path to image folder. The video will be created from these images.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=60,
        help='FPS (Frames per second) setting for the video.')
    args = parser.parse_args()

    video_file = args.image_folder + '.mp4'
    print("Creating video {}, FPS={}".format(video_file, args.fps))
    clip = ImageSequenceClip(args.image_folder, fps=args.fps)
    clip.write_videofile(video_file)


if __name__ == '__main__':
    main()


# Alternative

# #!/usr/local/bin/python3

# import cv2
# import argparse
# import os

# # Construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
# ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
# args = vars(ap.parse_args())

# # Arguments
# dir_path = 'run1'
# ext = args['extension']
# output = args['output']

# images = []
# for f in os.listdir(dir_path):
#     if f.endswith(ext):
#         images.append(f)

# # Determine the width and height from the first image
# image_path = os.path.join(dir_path, images[0])
# frame = cv2.imread(image_path)
# #cv2.imshow('video',frame)
# height, width, channels = frame.shape

# # Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
# out = cv2.VideoWriter(output, fourcc, 60.0, (width, height))

# for i in range(0,len(images),2):
#     print(str(i), end='\r', flush=True)

#     image_path = os.path.join(dir_path, images[i])
#     frame = cv2.imread(image_path)

#     out.write(frame) # Write out frame to video

#     #cv2.imshow('video',frame)
#     if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
#         break

# # Release everything if job is finished
# out.release()
# cv2.destroyAllWindows()

# print("The output video is {}".format(output))
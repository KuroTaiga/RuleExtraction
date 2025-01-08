import pyopenpose as op

params = {"model_folder": "models/", "video": "videos/input_video.mp4"}
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process video
datum = op.Datum()
opWrapper.emplaceAndPop([datum])

# Access keypoints
print(datum.poseKeypoints)

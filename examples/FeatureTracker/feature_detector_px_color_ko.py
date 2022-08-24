# RUN :                                      python3 feature_detector_px.py  
# RUN with graph node:   pipeline_graph run "python3 feature_detector_px.py"  
# DEBUG script :         DEPTHAI_LEVEL=debug python3 feature_detector_px.py 

#!/usr/bin/env python3

import cv2
import depthai as dai


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
featureTrackerLeft = pipeline.create(dai.node.FeatureTracker)

xoutPassthroughFrameLeft = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesLeft = pipeline.create(dai.node.XLinkOut)
xinTrackedFeaturesConfig = pipeline.create(dai.node.XLinkIn)

xoutPassthroughFrameLeft.setStreamName("passthroughFrameLeft")
xoutTrackedFeaturesLeft.setStreamName("trackedFeaturesLeft")
xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

# Properties
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)  
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
#camRgb.setFps(15)
camRgb.setIspScale(1, 2) # scale 0.5   to use  THE_400_P

# Disable optical flow
featureTrackerLeft.initialConfig.setMotionEstimator(False)

# Linking
#camRgb.video.link(featureTrackerLeft.inputImage)
camRgb.isp.link(featureTrackerLeft.inputImage)
featureTrackerLeft.passthroughInputImage.link(xoutPassthroughFrameLeft.input)
featureTrackerLeft.outputFeatures.link(xoutTrackedFeaturesLeft.input)
xinTrackedFeaturesConfig.out.link(featureTrackerLeft.inputConfig)

featureTrackerConfig = featureTrackerLeft.initialConfig.get()

print("Press 's' to switch between Harris and Shi-Thomasi corner detector!")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues used to receive the results
    passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 8, False)
    outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 8, False)

    #passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 1, True)
    #outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 1, True)

    inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig")

    leftWindowName = "left"

    def drawFeatures(frame, features):
        pointColor = (0, 0, 255)
        circleRadius = 2
        for feature in features:
            cv2.circle(frame, (int(feature.position.x), int(feature.position.y)), circleRadius, pointColor, -1, cv2.LINE_AA, 0)

    while True:
        inPassthroughFrameLeft = passthroughImageLeftQueue.get()
        passthroughFrameLeft = inPassthroughFrameLeft.getFrame()
        leftFrame = cv2.cvtColor(passthroughFrameLeft, cv2.COLOR_GRAY2BGR)

        trackedFeaturesLeft = outputFeaturesLeftQueue.get().trackedFeatures
        drawFeatures(leftFrame, trackedFeaturesLeft)

        # Show the frame
        cv2.imshow(leftWindowName, leftFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if featureTrackerConfig.cornerDetector.type == dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS:
                featureTrackerConfig.cornerDetector.type = dai.FeatureTrackerConfig.CornerDetector.Type.SHI_THOMASI
                print("Switching to Shi-Thomasi")
            else:
                featureTrackerConfig.cornerDetector.type = dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
                print("Switching to Harris")

            cfg = dai.FeatureTrackerConfig()
            cfg.set(featureTrackerConfig)
            inputFeatureTrackerConfigQueue.send(cfg)

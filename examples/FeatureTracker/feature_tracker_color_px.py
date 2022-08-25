"""
Using mono color camera
TODO: add roi maker node (ImageManip)
    In th script node
    1) at the beguining stock features marker detected in the init polygonal roi
    2) then tracking just these features
    3) update bounding box of the followed features 
    4) with this new bounding box the update roi (new pos roi=pre pos roi + movement bounding box ) 
"""
# RUN :                                      python3 feature_tracker_color_px.py  
# RUN with graph node:   pipeline_graph run "python3 feature_tracker_color_px.py"  
# DEBUG script :         DEPTHAI_LEVEL=debug python3 feature_tracker_color_px.py 

#!/usr/bin/env python3

#!/usr/bin/env python3

import cv2
import depthai as dai
from collections import deque

class FeatureTrackerDrawer:

    lineColor = (200, 0, 200)
    pointColor = (0, 0, 255)
    circleRadius = 2
    maxTrackedFeaturesPathLength = 30
    # for how many frames the feature is tracked
    trackedFeaturesPathLength = 10

    trackedIDs = None
    trackedFeaturesPath = None

    def onTrackBar(self, val):
        FeatureTrackerDrawer.trackedFeaturesPathLength = val
        pass

    def trackFeaturePath(self, features):

        newTrackedIDs = set()
        for currentFeature in features:
            currentID = currentFeature.id
            newTrackedIDs.add(currentID)

            if currentID not in self.trackedFeaturesPath:
                self.trackedFeaturesPath[currentID] = deque()

            path = self.trackedFeaturesPath[currentID]

            path.append(currentFeature.position)
            while(len(path) > max(1, FeatureTrackerDrawer.trackedFeaturesPathLength)):
                path.popleft()

            self.trackedFeaturesPath[currentID] = path

        featuresToRemove = set()
        for oldId in self.trackedIDs:
            if oldId not in newTrackedIDs:
                featuresToRemove.add(oldId)

        for id in featuresToRemove:
            self.trackedFeaturesPath.pop(id)

        self.trackedIDs = newTrackedIDs

    def drawFeatures(self, img):

        cv2.setTrackbarPos(self.trackbarName, self.windowName, FeatureTrackerDrawer.trackedFeaturesPathLength)

        for featurePath in self.trackedFeaturesPath.values():
            path = featurePath

            for j in range(len(path) - 1):
                src = (int(path[j].x), int(path[j].y))
                dst = (int(path[j + 1].x), int(path[j + 1].y))
                cv2.line(img, src, dst, self.lineColor, 1, cv2.LINE_AA, 0)
            j = len(path) - 1
            cv2.circle(img, (int(path[j].x), int(path[j].y)), self.circleRadius, self.pointColor, -1, cv2.LINE_AA, 0)

    def __init__(self, trackbarName, windowName):
        self.trackbarName = trackbarName
        self.windowName = windowName
        cv2.namedWindow(windowName)
        cv2.createTrackbar(trackbarName, windowName, FeatureTrackerDrawer.trackedFeaturesPathLength, FeatureTrackerDrawer.maxTrackedFeaturesPathLength, self.onTrackBar)
        self.trackedIDs = set()
        self.trackedFeaturesPath = dict()


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
colorCam = pipeline.create(dai.node.ColorCamera)
featureTrackerColor = pipeline.create(dai.node.FeatureTracker)

xoutPassthroughFrameColor = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesColor = pipeline.create(dai.node.XLinkOut)
xinTrackedFeaturesConfig = pipeline.create(dai.node.XLinkIn)




xoutPassthroughFrameColor.setStreamName("passthroughFrameColor")
xoutTrackedFeaturesColor.setStreamName("trackedFeaturesColor")
xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

# Properties
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
#colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)  #ko

if 1:
    colorCam.setIspScale(2,3)
    colorCam.video.link(featureTrackerColor.inputImage)
else:
    colorCam.isp.link(featureTrackerColor.inputImage)

# Linking
featureTrackerColor.passthroughInputImage.link(xoutPassthroughFrameColor.input)
featureTrackerColor.outputFeatures.link(xoutTrackedFeaturesColor.input)
xinTrackedFeaturesConfig.out.link(featureTrackerColor.inputConfig)

# By default the least mount of resources are allocated
# increasing it improves performance
numShaves = 2
numMemorySlices = 2
featureTrackerColor.setHardwareResources(numShaves, numMemorySlices)
featureTrackerConfig = featureTrackerColor.initialConfig.get()

print("Press 's' to switch between Lucas-Kanade optical flow and hardware accelerated motion estimation!")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues used to receive the results
    passthroughImageColorQueue = device.getOutputQueue("passthroughFrameColor", 8, False)
    outputFeaturesColorQueue = device.getOutputQueue("trackedFeaturesColor", 8, False)

    inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig")

    colorWindowName = "color"
    colorFeatureDrawer = FeatureTrackerDrawer("Feature tracking duration (frames)", colorWindowName)

    while True:
        inPassthroughFrameColor = passthroughImageColorQueue.get()
        passthroughFrameColor = inPassthroughFrameColor.getCvFrame()
        colorFrame = passthroughFrameColor

        trackedFeaturesColor = outputFeaturesColorQueue.get().trackedFeatures
        colorFeatureDrawer.trackFeaturePath(trackedFeaturesColor)
        colorFeatureDrawer.drawFeatures(colorFrame)

        # Show the frame
        cv2.imshow(colorWindowName, colorFrame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s'):
            if featureTrackerConfig.motionEstimator.type == dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW:
                featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.HW_MOTION_ESTIMATION
                print("Switching to hardware accelerated motion estimation")
            else:
                featureTrackerConfig.motionEstimator.type = dai.FeatureTrackerConfig.MotionEstimator.Type.LUCAS_KANADE_OPTICAL_FLOW
                print("Switching to Lucas-Kanade optical flow")

            cfg = dai.FeatureTrackerConfig()
            cfg.set(featureTrackerConfig)
            inputFeatureTrackerConfigQueue.send(cfg)

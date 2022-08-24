# RUN :                                      python3 feature_detector_px_left.py  
# RUN with graph node:   pipeline_graph run "python3 feature_detector_px_left.py"  
# DEBUG script :         DEPTHAI_LEVEL=debug python3 feature_detector_px_left.py 

#!/usr/bin/env python3

import cv2
import depthai as dai


# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
featureTrackerLeft = pipeline.create(dai.node.FeatureTracker)

xoutPassthroughFrameLeft = pipeline.create(dai.node.XLinkOut)
xoutTrackedFeaturesLeft = pipeline.create(dai.node.XLinkOut)
#xinTrackedFeaturesConfig = pipeline.create(dai.node.XLinkIn)

maxFrameSize = monoLeft.getResolutionHeight() * monoLeft.getResolutionWidth() * 3  #??

#Apply roi
manip1 = pipeline.create(dai.node.ImageManip)

#Simulation init marker roi
manip1.initialConfig.setCropRect(0, 0.7, 0.3, 1)
#manip1.initialConfig.setCropRect(0, 0.5, 0.5, 1)
#manip1.initialConfig.setCropRect(0, 0, 0.5, 1) #original
manip1.setMaxOutputFrameSize(maxFrameSize)
monoLeft.out.link(manip1.inputImage)

xoutPassthroughFrameLeft.setStreamName("passthroughFrameLeft")
xoutTrackedFeaturesLeft.setStreamName("trackedFeaturesLeft")
#xinTrackedFeaturesConfig.setStreamName("trackedFeaturesConfig")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

# Disable optical flow
featureTrackerLeft.initialConfig.setMotionEstimator(False)

# Linking
#monoLeft.out.link(featureTrackerLeft.inputImage)
manip1.out.link(featureTrackerLeft.inputImage)

featureTrackerLeft.passthroughInputImage.link(xoutPassthroughFrameLeft.input)
featureTrackerLeft.outputFeatures.link(xoutTrackedFeaturesLeft.input)
#xinTrackedFeaturesConfig.out.link(featureTrackerLeft.inputConfig)

featureTrackerConfig = featureTrackerLeft.initialConfig.get()

xout1 = pipeline.create(dai.node.XLinkOut)
xout1.setStreamName('out1')
manip1.out.link(xout1.input)

print("Press 's' to switch between Harris and Shi-Thomasi corner detector!")

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues used to receive the results
    passthroughImageLeftQueue = device.getOutputQueue("passthroughFrameLeft", 8, False)
    outputFeaturesLeftQueue = device.getOutputQueue("trackedFeaturesLeft", 8, False)

    q1 = device.getOutputQueue(name="out1", maxSize=4, blocking=False)
    
    #inputFeatureTrackerConfigQueue = device.getInputQueue("trackedFeaturesConfig")

    leftWindowName = "left"

    def drawFeatures(frame, features):
        pointColor = (0, 0, 255)
        circleRadius = 2
        for feature in features:
            cv2.circle(frame, (int(feature.position.x), int(feature.position.y)), circleRadius, pointColor, -1, cv2.LINE_AA, 0)
    
    def boundingFeatures(features):
        minX=100000
        minY=100000
        maxX=0
        maxY=0
        for feature in features:
            if(int(feature.position.x)<minX):
                minX=int(feature.position.x)
            if(int(feature.position.y)<minY):
                minY=int(feature.position.y)
            if(int(feature.position.x)>maxX):
                maxX=int(feature.position.x)
            if(int(feature.position.y)>maxY):
                maxY=int(feature.position.y)
        return minX,minY,maxX,maxY 

    while True:
        inPassthroughFrameLeft = passthroughImageLeftQueue.get()
        passthroughFrameLeft = inPassthroughFrameLeft.getFrame()
        leftFrame = cv2.cvtColor(passthroughFrameLeft, cv2.COLOR_GRAY2BGR)

        trackedFeaturesLeft = outputFeaturesLeftQueue.get().trackedFeatures
        drawFeatures(leftFrame, trackedFeaturesLeft)

        #TODO: set new roi from detected features
        minX,minY,maxX,maxY = boundingFeatures(trackedFeaturesLeft)
        print("features roi : "+str(minX)+" ; "+str(minY)+" ; "+str(maxX)+" ; "+str(maxY))

        if q1.has():
            cv2.imshow("Tile 1", q1.get().getCvFrame())

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

            #cfg = dai.FeatureTrackerConfig()
            #cfg.set(featureTrackerConfig)
            #inputFeatureTrackerConfigQueue.send(cfg)

# RUN :                                      python apriltag_rgb_px.py  
# RUN with graph node:   pipeline_graph run "python apriltag_rgb_px.py"  

#!/usr/bin/env python3

import cv2
import depthai as dai
import time

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
aprilTag = pipeline.create(dai.node.AprilTag)
manip = pipeline.create(dai.node.ImageManip)
image_manip_script = pipeline.create(dai.node.Script) #config to return luminances and corners
  
xoutAprilTag = pipeline.create(dai.node.XLinkOut)
xoutAprilTagImage = pipeline.create(dai.node.XLinkOut)

xoutAprilTag.setStreamName("aprilTagData")
xoutAprilTagImage.setStreamName("aprilTagImage")

# Properties
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P) # FPS
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) #12 FPS
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)     # 7 FPS  
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

manip.initialConfig.setResize(480, 270)
manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

aprilTag.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)

# Linking
aprilTag.passthroughInputImage.link(xoutAprilTagImage.input)
camRgb.video.link(manip.inputImage)
manip.out.link(aprilTag.inputImage)
aprilTag.out.link(xoutAprilTag.input)

aprilTag.out.link(image_manip_script.inputs['aprilTagData'])  

# always take the latest frame as apriltag detections are slow
aprilTag.inputImage.setBlocking(False)
aprilTag.inputImage.setQueueSize(1)

# advanced settings, configurable at runtime
aprilTagConfig = aprilTag.initialConfig.get()
aprilTagConfig.quadDecimate = 4
aprilTagConfig.quadSigma = 0
aprilTagConfig.refineEdges = True
aprilTagConfig.decodeSharpening = 0.25
aprilTagConfig.maxHammingDistance = 1
aprilTagConfig.quadThresholds.minClusterPixels = 5
aprilTagConfig.quadThresholds.maxNmaxima = 10
aprilTagConfig.quadThresholds.criticalDegree = 10
aprilTagConfig.quadThresholds.maxLineFitMse = 10
aprilTagConfig.quadThresholds.minWhiteBlackDiff = 5
aprilTagConfig.quadThresholds.deglitch = False
aprilTag.initialConfig.set(aprilTagConfig)



image_manip_script.setScript("""
    import time
    
    cfg = ImageManipConfig()
    while True:
        time.sleep(0.1) # Avoid lazy looping

        aprilTagData = node.io['aprilTagData'].tryGet()
        cornersReady=(aprilTagData is not None)
        node.warn(f"cornersReady {cornersReady}")
        if cornersReady:
             node.warn("d2")
             for i, aprilTag in enumerate(aprilTagData.aprilTags):
                node.warn("d4")
                topLeft = aprilTag.topLeft
                topRight = aprilTag.topRight
                bottomRight = aprilTag.bottomRight
                bottomLeft = aprilTag.bottomLeft
                
                node.warn(f"d51 topLeftX    {str(topLeft.x)}, Y {str(topLeft.y)}")
                node.warn(f"d52 topRight    {str(topRight.x)}, Y {str(topRight.y)}")
                node.warn(f"d53 bottomRight {str(bottomRight.x)}, Y {str(bottomRight.y)}")
                node.warn(f"d54 bottomLeft  {str(bottomLeft.x)}, Y {str(bottomLeft.y)}")
                
                cfg.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
                config.setResize(320,320)
                node.io['manip_cfg'].send(cfg)
                   
    """)


end_xout = pipeline.create(dai.node.XLinkOut)
end_xout.setStreamName("end")
image_manip_script.outputs['manip_cfg'].link(end_xout.input)




#Add ROI manip
roi_manip = pipeline.create(dai.node.ImageManip)
#roi_manip.setWaitForConfigInput(True)
roi_manip.initialConfig.setResize(62, 62)
image_manip_script.outputs['manip_cfg'].link(roi_manip.inputConfig)

#manip.out.link(roi_manip.inputImage)
aprilTag.passthroughInputImage.link(roi_manip.inputImage)

roi_xout = pipeline.create(dai.node.XLinkOut)
roi_xout.setStreamName("roi")
roi_manip.out.link(roi_xout.input)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    manipQueue = device.getOutputQueue("aprilTagImage", 8, False)
    aprilTagQueue = device.getOutputQueue("aprilTagData", 8, False)
    #manipQueue = device.getOutputQueue("aprilTagImage", 1, True)
    #aprilTagQueue = device.getOutputQueue("aprilTagData", 1, True)
    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while(True):
        inFrame = manipQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        monoFrame = inFrame.getFrame()
        frame = cv2.cvtColor(monoFrame, cv2.COLOR_GRAY2BGR)

        aprilTagData = aprilTagQueue.get().aprilTags
        for aprilTag in aprilTagData:
            topLeft = aprilTag.topLeft
            topRight = aprilTag.topRight
            bottomRight = aprilTag.bottomRight
            bottomLeft = aprilTag.bottomLeft

            center = (int((topLeft.x + bottomRight.x) / 2), int((topLeft.y + bottomRight.y) / 2))

            cv2.line(frame, (int(topLeft.x), int(topLeft.y)), (int(topRight.x), int(topRight.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(topRight.x), int(topRight.y)), (int(bottomRight.x), int(bottomRight.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomRight.x), int(bottomRight.y)), (int(bottomLeft.x), int(bottomLeft.y)), color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, (int(bottomLeft.x), int(bottomLeft.y)), (int(topLeft.x), int(topLeft.y)), color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(aprilTag.id)
            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        cv2.putText(frame, "Fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255,255,255))

        cv2.imshow("April tag frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

# RUN :                                      python3 apriltag_rgb_px.py  
# RUN with graph node:   pipeline_graph run "python3 apriltag_rgb_px.py"  
# DEBUG script :         DEPTHAI_LEVEL=debug python3 apriltag_rgb_px.py 

import cv2
import depthai as dai
import time

#from MultiMsgSync import TwoStageHostSeqSync

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
aprilTag = pipeline.create(dai.node.AprilTag)
manip = pipeline.create(dai.node.ImageManip)
# Script node will take the output from the AprilTag node as an input and set ImageManipConfig
# to the 'roi_manip' to crop the initial frame
image_manip_script = pipeline.create(dai.node.Script) #config to return luminances and corners
    
xoutAprilTag = pipeline.create(dai.node.XLinkOut)
xoutAprilTagImage = pipeline.create(dai.node.XLinkOut)

xoutAprilTag.setStreamName("aprilTagData") #detected marker corners 
xoutAprilTagImage.setStreamName("aprilTagImage")    #low resolution image

# Properties
#camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P) #12 FPS
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)     # 7 FPS  
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)

manip.initialConfig.setResize(480, 270)
#manip.initialConfig.setResize(480, 270) # px ?
manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)

aprilTag.initialConfig.setFamily(dai.AprilTagConfig.Family.TAG_36H11)

# Linking
aprilTag.passthroughInputImage.link(xoutAprilTagImage.input)
camRgb.video.link(manip.inputImage)
manip.out.link(aprilTag.inputImage)
aprilTag.out.link(xoutAprilTag.input) # #detected marker corners 

camRgb.video.link(image_manip_script.inputs['video'])
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
    msgs = dict()

    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")

        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg

        # To avoid freezing (not necessary for this ObjDet model)
        if 15 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element

    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")

            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None

    def correct_bb(bb):
        if bb.xmin < 0: bb.xmin = 0.001
        if bb.ymin < 0: bb.ymin = 0.001
        if bb.xmax > 1: bb.xmax = 0.999
        if bb.ymax > 1: bb.ymax = 0.999
        return bb

    while True:
        time.sleep(0.001) # Avoid lazy looping

        video = node.io['video'].tryGet()
        aprilTagData = node.io['aprilTagData'].tryGet()
        
        node.warn("d1")
        frameReady=(video is not None)
        cornersReady=(aprilTagData is not None)
        dataReady= frameReady and cornersReady
        node.warn(f"frameReady {frameReady}, cornersReady {cornersReady}, dataReady {dataReady}")
        if dataReady:
             node.warn("debugTmp4")        
    """)

roi_manip = pipeline.create(dai.node.ImageManip)
#roi_manip.initialConfig.setResize(62, 62)
roi_manip.setFrameType(dai.ImgFrame.Type.GRAY8) #TODO convert luminance
roi_manip.setWaitForConfigInput(True)
image_manip_script.outputs['manip_cfg'].link(roi_manip.inputConfig)
image_manip_script.outputs['manip_img'].link(roi_manip.inputImage)

roi_xout = pipeline.create(dai.node.XLinkOut)
roi_xout.setStreamName("roi")
roi_manip.out.link(roi_xout.input)



# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the mono frames from the outputs defined above
    manipQueue = device.getOutputQueue("aprilTagImage", 8, False)
    #aprilTagQueue = device.getOutputQueue("aprilTagData", 8, False) #detected marker corners 
    roimanipQueue = device.getOutputQueue("roi", 8, False)
    
    color = (0, 255, 0)

    startTime = time.monotonic()
    counter = 0
    fps = 0

    while(True):
        inFrame = manipQueue.get()
        #inFrameRoi = roimanipQueue.get()

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        monoFrame = inFrame.getFrame()
        #roimonoFrame = inFrameRoi.getFrame()
        
        cv2.imshow("monoFrame", monoFrame)
        #cv2.imshow("April tag frame roi ", roiframe)

        if cv2.waitKey(1) == ord('q'):
            break


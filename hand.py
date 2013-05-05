#!/usr/bin/python

import cv2
import cv
import numpy
from PIL import Image
from math import fabs

VIDEO_FILE = "/tmp/ramdisk/video.avi"
VIDEO_FORMAT = cv.CV_FOURCC('I', 'Y', 'U', 'V')
NUM_FINGERS = 5
NUM_DEFECTS = 8
SHOW_HAND_CONTOUR = 1


class ctx:
   ' object to hold stuff'
   empCount = False
   writer = False

   image = False
   thr_image = False
   temp_image1 = False
   temp_image3 = False

   contour = False
   hull = False
   hand_center = False
   fingers = False
   defects = False

   hull_st = False
   contour_st = False
   temp_st = False
   defects_st = False

   kernel = False

   num_fingers = False
   hand_radius = False
   num_defects = False


def init_capture(ctx):
	ctx.capture = cv2.VideoCapture(0)

	if not ctx.capture:
		print "Error initializing Capture"
		sys.exit(1)

	rval,ctx.image = ctx.capture.read() 
	if not rval:
		print "cannot read camera"
		sys.exit(1)
	else:
		ctx.image = cv.fromarray(ctx.image)



def init_recording(ctx):

	fps = ctx.capture.get(cv.CV_CAP_PROP_FPS)
	width = int(ctx.capture.get(cv.CV_CAP_PROP_FRAME_WIDTH))
	height = int(ctx.capture.get(cv.CV_CAP_PROP_FRAME_HEIGHT))


	if fps < 10:
		fps = 10


	ctx.writer = cv2.VideoWriter(VIDEO_FILE, VIDEO_FORMAT, fps, (width, height), 1)

	if not ctx.writer.isOpened():
		print "cannot capture video"
		sys.exit(1)


def init_windows():
	cv.NamedWindow("output", cv.CV_WINDOW_AUTOSIZE)
	cv.NamedWindow("thresholded", cv.CV_WINDOW_AUTOSIZE)
	cv.MoveWindow("output", 50, 50)
	cv.MoveWindow("thresholded", 700, 50)





def init_ctx(ctx):

	ctx.thr_image = cv.CreateImage(cv.GetSize(ctx.image), 8, 1)
	ctx.thr_image = numpy.asarray(ctx.thr_image[:,:])
	ctx.temp_image1 = cv.CreateImage(cv.GetSize(ctx.image), 8, 1)
	ctx.temp_image1 = numpy.asarray(ctx.temp_image1[:,:])

	ctx.temp_image3 = cv.CreateImage(cv.GetSize(ctx.image), 8, 3)
	ctx.temp_image3 = numpy.asarray(ctx.temp_image3[:,:])
	ctx.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9), (4,4))


	ctx.contour_st = cv.CreateMemStorage(0)
	ctx.hull_st = cv.CreateMemStorage(0)
	ctx.temp_st = cv.CreateMemStorage(0)
	#ctx.fingers = calloc(NUM_FINGERS + 1, sizeof(cv.CvPoint));
	#ctx.defects = calloc(NUM_DEFECTS, sizeof(cv.CvPoint));

def filter_and_threshold(ctx):
	# Soften image 
	cv2.GaussianBlur(ctx.image, (11, 11), 0, ctx.temp_image3) 
	#cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])  dst

	#cv.Smooth(ctx.image, ctx.temp_image3, cv.CV_GAUSSIAN, 11, 11, 0, 0);
	#cv.Smooth(src, dst, smoothtype=CV_GAUSSIAN, param1=3, param2=0, param3=0, param4=0)  None
	# Remove some impulsive noise 
	cv2.medianBlur(ctx.temp_image3, 11, ctx.temp_image3)
	#cv2.medianBlur(src, ksize[, dst])  dst
	#cv.Smooth(ctx.temp_image3, ctx.temp_image3, cv.CV_MEDIAN, 11, 11, 0, 0)
	cv2.cvtColor(ctx.temp_image3, cv.CV_BGR2HSV, ctx.temp_image3 ) 
	#cv.CvtColor(ctx.temp_image3, ctx.temp_image3, cv.CV_BGR2HSV)

	#ctx.temp_image3 = toNumpy(ctx.temp_image3)
	cv2.inRange(ctx.temp_image3, cv.Scalar(0, 0, 160, 0), cv.Scalar(255, 400, 300, 255), ctx.thr_image)

	# Apply morphological opening 
	ctx.thr_image = cv2.morphologyEx(ctx.thr_image, cv.CV_MOP_OPEN, ctx.kernel)

	#ctx.thr_image = cv.fromarray(ctx.thr_image)
	#cv.Smooth(ctx.thr_image, ctx.thr_image, cv.CV_GAUSSIAN, 3, 3, 0, 0)
	cv2.GaussianBlur(ctx.thr_image, (3, 3), 0, ctx.thr_image) 


def toNumpy(img):
	return numpy.asarray(img[:,:])

def toIpl(img):
	return cv.fromarray(img)

def find_contour(ctx):
	contour = False
	max_area = 0.0
	ctx.temp_image1 = toIpl(ctx.temp_image1)
	ctx.thr_image = toIpl(ctx.thr_image)

	cv.Copy(ctx.thr_image, ctx.temp_image1)
	ctx.temp_image1 = toNumpy(ctx.temp_image1)
	contours, hierarchy = cv2.findContours(ctx.temp_image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	#cv.FindContours(ctx.temp_image1, ctx.temp_st, contours, sizeof(cv.Contour), cv.CV_RETR_EXTERNAL, cv.CV_CHAIN_APPROX_SIMPLE, cv.cvPoint(0, 0));
	
	#cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])  contours, hierarchy
	#cv.FindContours(image, storage, mode=CV_RETR_LIST, method=CV_CHAIN_APPROX_SIMPLE, offset=(0, 0))  contours


	for tmp in contours:
		area = fabs(cv2.contourArea(tmp))

		if area > max_area:
			max_area = area
			contour = tmp

	'''this doesnt run '''

	if type(contour) != 'bool':
		print type(contour)
		#Python: cv2.approxPolyDP(curve, epsilon, closed[, approxCurve])  approxCurve
		#C: CvSeq* cvApproxPoly(const void* src_seq, int header_size, CvMemStorage* storage, int method, double eps, int recursive=0 )
		#contour = cv.cvApproxPoly(contour, sizeof(cv.CvContour), ctx.contour_st, cv.CV_POLY_APPROX_DP, 2, 1)
		try:
			cv2.approxPolyDP(contour, 2, False, ctx.contour)
		except ValueError:
			print 'error, unknown'



def find_convex_hull(ctx):
	defects, defect_array = False, False
	i,x,y,dist = 0,0,0,0
	ctx.hull = False


	if type(ctx.contour) != 'numpy.ndarray':
		return

	#ctx.hull = cv.cvConvexHull2(ctx.contour, ctx.hull_st, cv.CV_CLOCKWISE, 0)
	cv2.convexHull(ctx.contour, ctx.hull_st, 1, ctx.hull)

	#Python: cv2.convexHull(points[, hull[, clockwise[, returnPoints]]])  hull
	#C: CvSeq* cvConvexHull2(const CvArr* input, void* hull_storage=NULL, int orientation=CV_CLOCKWISE, int return_points=0 )


	if ctx.hull:
		ctx.contour = toIpl(ctx.contour)
		defects = cv2.cv.ConvexityDefects(ctx.contour, ctx.hull, ctx.defects_st)
		#defects = cv.cvConvexityDefects(ctx.contour, ctx.hull, ctx.defects_st)

		#C++: void convexityDefects(InputArray contour, InputArray convexhull, OutputArray convexityDefects)
		#Python: cv2.convexityDefects(contour, convexhull[, convexityDefects])  convexityDefects

		if defects and defects.total:
			defect_array = calloc(defects.total, sizeof(cv.cvConvexityDefect))
			cv.cvCvtSeqToArray(defects, defect_array, cv.CV_WHOLE_SEQ)

			for i in NUM_DEFECTS:
				x += i.depth_point.x
				y += i.depth_point.y
				ctx.defects[i] = cv.cvPoint(i.depth_point.x, i.depth_point.y)

			x /= defects.total
			y /= defects.total
		ctx.num_defects = defects.total
		ctx.hand_center = cv.cvPoint(x,y)

		for i in defects.total:
			d = (x - i.depth_point.x) * (x - i.depth_point.x) + (y - i.depth_point.y) * (y - i.depth_point.y)
			dist += sqrt(d)

		ctx.hand_radius = dist / defects.total
		del defect_array


def find_fingers(ctx):
	n,i,points,max_point,dist1,dist2,finger_distance = 0,0,0,0,0,0,0

	ctx.num_fingers = 0

	if not type(ctx.contour) == 'bool':	
		return False

	if not type(ctx.hull) == 'bool':
		return False

	print  type(ctx.contour)
	print  type(ctx.hull)


	for points in  ctx.contour:
		dist = False
		cx = ctx.hand_center.x
		cy = ctx.hand_center.y

		dist = (cx - points[i].x) * (cx - points[i].x) + (cy - points[i].y) * (cy - points[i].y)
		if (dist < dist1 and dist1 > dist2 and max_point.x != 0 and max_point.y < cv.cvGetSize(ctx.image).height - 10):
			ctx.num_fingers=ctx.num_fingers+1
			finger_distance[ctx.num_fingers] = max_point
			if (ctx.num_fingers >= NUM_FINGERS + 1):
				break

		dist2 = dist1
		dist1 = dist
		max_point = points[i]
	del points

def display(ctx):
	i = False

	if ctx.num_fingers == NUM_FINGERS:
		if SHOW_HAND_CONTOUR:
			cv.cvDrawContours(ctx.image, ctx.contour, cv.CV_RGB(0,0,255), cv.CV_RGB(0,255,0), 0, 1, cv.CV_AA, cv.cvPoint(0,0))
		cv.cvCircle(ctx.image, ctx.hand_center, 5, cv.CV_RGB(255,0,255), 1, CV_AA,0)
		cv.cvCircle(ctx.image, ctx.hand_center, ctx.hand_radius, cv.CV_RGB(255,0,0), 1, CV_AA, 0)

		for i in ctx.num_fingers:
			cv.cvCircle(ctx.image, ctx.fingers[i], 10, cv.CV_RGB(0,255,0), 3, cv.CV_AA, 0)
			cv.cvLine(ctx.image, ctx.hand_center, ctx.fingers[i], cv.CV_RGB(255,255,0), 1, cv.CV_AA, 0)

		for i in ctx.num_defects:
			cv.cvCircle(ctx.image, ctx.defects[i], 2, cv.CV_RGB(200,200,200), 2, cv.CV_AA, 0)

	ctx.image = toNumpy(ctx.image)
	cv2.imshow("output", ctx.image)
	ctx.thr_image = toNumpy(ctx.thr_image)
	cv2.imshow("thresholded", ctx.thr_image)



if __name__ == '__main__':
	ctx = ctx()
	key = False

	init_capture(ctx)
	init_recording(ctx)
	init_windows()
	init_ctx(ctx)

	while key != 'q':
		#ctx.image = cv.QueryFrame(ctx.capture)
		rval,ctx.image = ctx.capture.read() 
		if not rval:
			print "cannot read camera"
			sys.exit(1)


		filter_and_threshold(ctx)
		find_contour(ctx)
		find_convex_hull(ctx)
		find_fingers(ctx)

		display(ctx)

		ctx.writer.write(ctx.image)

		key = cv2.waitKey(1)
		#
#	return False
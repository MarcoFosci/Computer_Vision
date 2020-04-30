import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import keras.backend as K 

def create_dataset(n_imgs, n_bb, side_dim, obj_min_dim, obj_max_dim, min_threshold, max_threshold):
  # creation of objects in images and relative bounding boxes
  '''  
  Arguments:
  N_imgs -- scalar, number of images
  N_bb -- scalar, number of Bounding Boxes
  side_dim -- size of image sides
  obj_min_dim -- scalar, minimum object size
  obj_max_dim -- scalar, maximum object size
  min_threshold - scalar, minimum color threshold
  max_threshold - scalar, maximum color threshold
  
  Returns:
  imgs -- np.array(N_imgs, side_dim, side_dim), images of our dataset 
  bounding_boxes -- np.array(N_imgs, N_bb, 5), bounding boxes for each image
  distance -- np.array(N_imgs, N_bb), distance from origin for each Bounding box
  '''
  for N_img in range(n_imgs):
    for i_ogg in range(N_bb):
      w = np.random.randint(obj_min_dim, obj_max_dim)
      h = np.random.randint(obj_min_dim, obj_max_dim)
      x = np.random.randint(0, side_dim - w)
      y = np.random.randint(0, side_dim - h)
          
      R = (np.random.randint(min_threshold, max_threshold)) / 255
      G = (np.random.randint(min_threshold, max_threshold)) / 255
      B = (np.random.randint(min_threshold, max_threshold)) / 255
        
      imgs[N_img, y:y+h, x:x+w, 0] = R 
      imgs[N_img, y:y+h, x:x+w, 1] = G 
      imgs[N_img, y:y+h, x:x+w, 2] = B 
        

      Pc = 1. # for the moment we use for all target value Pc = 1
      bounding_boxes[N_img, i_ogg] = [Pc, x, y, w, h]   
        
        # we calculate the distance from the origin for each Bounding Box
      distance[N_img, i_ogg] = np.sqrt(np.square(x+(w/2))+ np.square(y+(h/2)))
  return imgs, bounding_boxes, distance


def order_bb(bounding_boxes, distance):
  # bounding boxes sorting based on distance from the origin
  '''  
  Arguments:
  bounding_boxes -- np.array(N_imgs, N_bb, 5), bounding boxes for each image
  distance -- np.array(N_imgs, N_bb), distance from origin for each Bounding box

  Returns:
  bboxes -- np.array(N_imgs, N_bb, 5), bounding boxes ordered 
  '''
  
  n_imgs = bounding_boxes.shape[0]
  n_bb = bounding_boxes.shape[1]
  
  order = np.zeros((n_imgs, n_bb), dtype=int)
  bboxes = np.zeros(bounding_boxes.shape)

  for i in range(N_imgs):
    order[i] = np.argsort(distance[i]) 
    for j in range(N_bb):
      bboxes[i,j] = (bounding_boxes[i, order[i,j]])
      
  return bboxes

def euc_loss(y_true, y_pred):
  '''  
  Arguments:
  y_true -- tensor with expected Y
  y_pred -- tensor with predicted Y
  
  Returns:
  euc_dist -- scalar
  '''
  euc_dist = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
  return euc_dist

def rmse(y_true, y_pred):
  '''  
  Arguments:
  y_true -- tensor with expected Y
  y_pred -- tensor with predicted Y
  
  Returns:
  euc_dist -- scalar
  '''
  rmse = K.sqrt(K.mean(K.square(y_pred -y_true), axis=-1))
  return rmse

# We define one of the reference metrics to analyze our results
# the Intersection over Union that measures the correspondence 
# of our predictions with the actual bounding boxes

def IoU(box1, box2):
  '''  
  Arguments:
  box1 -- first box, with coordinates (x, y, w, h)
  box2 -- second box, with coordinates (x2, y2, w2, h2)
  
  Returns:
  iou -- scalar
  '''    
  [x, y, w, h] = box1
  [x2, y2, w2, h2] = box2
    
  # Intesection area 
  xi1 = np.maximum(x, x2)
  yi1 = np.maximum(y, y2)
  xi2 = np.minimum(x+w, x2+w2)
  yi2 = np.minimum(y+h, y2+h2)
  xi = np.maximum(0., xi2-xi1)
  yi = np.maximum(0., yi2-yi1)
  
  inter_area = np.multiply(xi,yi)

  # Union area
  box1_area = (w*h)
  box2_area = (w2*h2)
  union_area = box1_area+box2_area-inter_area
  
  # Calculation of IoU
  iou = inter_area/union_area

  return iou

# The average IoU of the a whole dataset 
def mean_IoU(expe_bboxes, pred_bboxes):
  '''  
  Arguments:
  expe_bboxes -- np.array(dim_test, N_bb, 5), expected bounding boxes 
  pred_bboxes -- np.array(dim_test, N_bb, 5), predicted bounding boxes 
  
  Returns:
  iou -- scalar, value of Intersection over Union of the whole dataset
  '''    
  
  [n_examples, n_bb] = expe_bboxes.shape[:2] 
  IoU_test = np.zeros((n_examples, N_bb)) 
  iou_global = 0

  for i in range(n_examples):
    for j in range(n_bb):

      [x , y, w, h] = pred_bboxes[i, j, 1:] 
      [x2, y2, w2, h2] = expe_bboxes[i, j, 1:] 

      IoU_test[i,j] = IoU([x,y,w,h], [x2,y2,w2,h2])

  iou_global = np.mean(IoU_test)
  return iou_global

def visualize(img, dist, bboxes, expe_bboxes, iou_tr):  
  '''  
  Arguments:
  img -- np.array(32,32,3), an image
  dist -- np.array(4), objects distance from origin
  bboxes -- np.array(4,5), predicted bounding boxes 
  expe_bboxes -- np.array(4,5), expected bounding boxes 
  iou_tr -- scalar, IoU treshold
  
  Returns:
  None
  ''' 
  side_dim = img.shape[0]

  # We look at the objects contained in it...
  ax.imshow(img, origin = 'upper', extent=[0, side_dim, side_dim, 0])
  
  if np.any(bboxes != None) and np.any(dist == None):
    n_bb = bboxes.shape[0]
  
    # ...and the bounding boxes predicted by the algorithm
    for j in range (n_bb):
      [Pc, x , y, w, h] = bboxes[j] 
      
      if np.any(expe_bboxes != None): 
        [x2, y2, w2, h2] = expe_bboxes[j, 1:] 
        iou_bb = IoU([x,y,w,h], [x2,y2,w2,h2])
      
        if iou_bb >= iou_tr:
          rect = patches.Rectangle((x, y) , w, h, ec='tab:red', lw='1.4', fc='none')
          dida = 'IoU: ' + '{:0.3f}'.format(iou_bb)
          plt.annotate(dida, (x + w , y + 0.8), color='tab:red')
  
        else: 
          rect = patches.Rectangle((x, y) , w, h, ec='tab:blue', lw='1.4', fc='none')
        
        print('| Image', '{:5d}'.format(im), 'Object', j+1,'|', ' Confidence:', '{:0.3f}'.format(Pc), '|--- BBox IoU:', '{0:2.1%}'.format(iou_bb),'---')
        # print('| B-box coordinates: ', '[{:0.1f}'.format(x), '{:0.1f}'.format(y), '{:0.1f}'.format(w), '{:0.1f}]'.format(h))
        # print('| Object coordinates:', '[{:0.1f}'.format(x2), '{:0.1f}'.format(y2), '{:0.1f}'.format(w2), '{:0.1f}]'.format(h2))
    
      else:
        rect = patches.Rectangle((x, y) , w, h, ec='r', lw='1.4', fc='none')
        
      ax.add_patch(rect)
  elif np.any(bboxes != None) and np.any(dist != None):
    n_bb = bboxes.shape[0]
    for j in range (n_bb):
      [_, x, y, w, h] = bboxes[j]
      line = patches.Arrow(0, 0, x+(w/2), y+(h/2), .3, ec='r') 
      ax.add_patch(line)
      dida = 'dist: ' + '{:0.1f}'.format(dist[j])
      plt.annotate(dida, (x + w + 0.8 , y + 0.8), color='r')

  

def visual_metric(train_ex, m_iou, N_ep, Batchs, history, metric):
  print('Datatset mean IoU:', '{0:2%}'.format(m_iou))
  print('Final train loss:', '{0:5}'.format(history['val_loss'][-1]))
  print('Final val loss:', '{0:5}'.format(history['loss'][-1]))
  print('Dim. train:', train_ex, 'examples')
  print('Epochs:', N_ep, '- mini-batches:', Batchs)

  # Plot training & validation accuracy values
  plt.plot(history[metric])
  plt.plot(history['val_' + metric])
  plt.title('Model ' + metric)
  plt.ylabel(metric)
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Validation'], loc='upper left')
  plt.show()

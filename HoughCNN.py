import caffe
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import DataManager as DM
import utilities
import cPickle as pkl
from os.path import splitext
from multiprocessing import Process, Queue
from sklearn.neighbors import NearestNeighbors
import scipy


EPS = 0.0000000001


class HoughCNN(object):
    params = None
    dataManagerTrain = None
    dataManagerTest = None

    def __init__(self,params):
        self.params=params
        caffe.set_device(self.params['ModelParams']['device'])
        caffe.set_mode_gpu()

    def prepareDataThread(self, dataQueue, numpyImages, numpyGT):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']

        keysIMG = numpyImages.keys()

        nr_iter_dataAug = nr_iter*batchsize
        np.random.seed()

        h_patch_size = int(self.params['ModelParams']['patchSize']/2)
        whichImageList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        np.random.rand()

        # todo: change in order to pick half/half foreground background
        whichCoordinateList_x = np.random.randint(low=h_patch_size + 2,
                                                  high=self.params['DataManagerParams']['VolSize'][0] - h_patch_size -2,
                                                  size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))
        whichCoordinateList_y = np.random.randint(low=h_patch_size + 2,
                                                  high=self.params['DataManagerParams']['VolSize'][1] - h_patch_size - 2,
                                                  size=int(nr_iter_dataAug / self.params['ModelParams']['nProc']))
        whichCoordinateList_z = np.random.randint(low=h_patch_size + 2,
                                                  high=self.params['DataManagerParams']['VolSize'][2] - h_patch_size - 2,
                                                  size=int(nr_iter_dataAug / self.params['ModelParams']['nProc']))
        whichCoordinateList = np.vstack((whichCoordinateList_x, whichCoordinateList_y, whichCoordinateList_z)).T

        whichDataForMatchingList = np.random.randint(len(keysIMG), size=int(nr_iter_dataAug/self.params['ModelParams']['nProc']))

        assert len(whichCoordinateList) == len(whichDataForMatchingList) == len(whichImageList)

        for whichImage, whichCoordinate, whichDataForMatching in \
                zip(whichImageList, whichCoordinateList, whichDataForMatchingList):
            filename, ext = splitext(keysIMG[whichImage])

            currGtKey = filename + '_segmentation' + ext
            currImgKey = filename + ext

            # data agugumentation through hist matching across different examples...
            ImgKeyMatching = keysIMG[whichDataForMatching]

            img = numpyImages[currImgKey]
            lab = numpyGT[currGtKey]

            img = utilities.hist_match(img, numpyImages[ImgKeyMatching]) #potentially inefficient (slow)
            imgPatch = img[whichCoordinate[0]-h_patch_size-1:whichCoordinate[0]+h_patch_size,
                       whichCoordinate[1] - h_patch_size - 1:whichCoordinate[1] + h_patch_size,
                       whichCoordinate[2] - h_patch_size - 1:whichCoordinate[2] + h_patch_size]

            imgPatchLab = lab[whichCoordinate[0]-h_patch_size-1:whichCoordinate[0]+h_patch_size,
                       whichCoordinate[1] - h_patch_size - 1:whichCoordinate[1] + h_patch_size,
                       whichCoordinate[2] - h_patch_size - 1:whichCoordinate[2] + h_patch_size]

            dataQueue.put(tuple((imgPatch, imgPatchLab)))

    def trainThread(self,dataQueue,solver):

        nr_iter = self.params['ModelParams']['numIterations']
        batchsize = self.params['ModelParams']['batchsize']
        h_p = int(self.params['ModelParams']['patchSize'] / 2)
        batchData = np.zeros((batchsize, 1,
                              self.params['ModelParams']['patchSize'],
                              self.params['ModelParams']['patchSize'],
                              self.params['ModelParams']['patchSize']), dtype=float)
        batchLabel = np.zeros((batchsize, 1), dtype=float)
        batchweight = np.zeros((batchsize, 1), dtype=float)

        train_loss = np.zeros(nr_iter)
        for it in range(nr_iter):
            for i in range(batchsize):
                [patch, label] = dataQueue.get()

                batchData[i, 0, :, :, :] = patch.astype(dtype=np.float32)
                batchLabel[i, 0] = label[h_p, h_p, h_p] > 0.5

            batchweight[batchLabel[:, 0] == 0, 0] = 1.0 / sum((batchLabel[:, 0] == 0).astype(dtype=float))
            batchweight[batchLabel[:, 0] == 1, 0] = 1.0 / sum((batchLabel[:, 0] == 1).astype(dtype=float))

            solver.net.blobs['data'].data[...] = batchData.astype(dtype=np.float32)
            solver.net.blobs['label'].data[...] = batchLabel.astype(dtype=np.float32)
            solver.net.blobs['weight'].data[...] = batchweight.astype(dtype=np.float32)
            #use only if you do softmax with loss

            solver.step(1)  # this does the training
            train_loss[it] = solver.net.blobs['loss'].data

            if (np.mod(it, 10) == 0):
                plt.clf()
                plt.plot(range(0, it), train_loss[0:it])
                plt.pause(0.00000001)

            matplotlib.pyplot.show()

    def train(self):
        print self.params['ModelParams']['dirTrain']

        #we define here a data manage object
        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData() #loads in sitk format

        howManyImages = len(self.dataManagerTrain.sitkImages)
        howManyGT = len(self.dataManagerTrain.sitkGT)

        assert howManyGT == howManyImages

        print "The dataset has shape: data - " + str(howManyImages) + ". labels - " + str(howManyGT)

        # Write a temporary solver text file because pycaffe is stupid
        if self.params['ModelParams']['solver'] is None:

            with open("solver.prototxt", 'w') as f:
                f.write("net: \"" + self.params['ModelParams']['prototxtTrain'] + "\" \n")
                f.write("base_lr: " + str(self.params['ModelParams']['baseLR']) + " \n")
                f.write("momentum: 0.99 \n")
                f.write("weight_decay: 0.0005 \n")
                f.write("lr_policy: \"step\" \n")
                f.write("stepsize: 20000 \n")
                f.write("gamma: 0.1 \n")
                f.write("display: 1 \n")
                f.write("snapshot: 500 \n")
                f.write("snapshot_prefix: \"" + self.params['ModelParams']['dirSnapshots'] + "\" \n")
                #f.write("test_iter: 3 \n")
                #f.write("test_interval: " + str(test_interval) + "\n")

            f.close()
            solver = caffe.SGDSolver("solver.prototxt")
            os.remove("solver.prototxt")
        else:
            solver = caffe.SGDSolver(self.params['ModelParams']['solver'])

        if (self.params['ModelParams']['snapshot'] > 0):
            solver.restore(self.params['ModelParams']['dirSnapshots'] + "_iter_" + str(
                self.params['ModelParams']['snapshot']) + ".solverstate")

        plt.ion()

        numpyImages = self.dataManagerTrain.getNumpyImages()
        numpyGT = self.dataManagerTrain.getNumpyGT()

        #numpyImages['Case00.mhd']
        #numpy images is a dictionary that you index in this way (with filenames)

        for ii, key in enumerate(numpyImages):
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])

            numpyImages[key]-=mean
            numpyImages[key]/=std

        dataQueue = Queue(250) #max 250 patches in queue
        dataPreparation = [None] * self.params['ModelParams']['nProc']

        #thread creation
        for proc in range(0,self.params['ModelParams']['nProc']):
            dataPreparation[proc] = Process(target=self.prepareDataThread, args=(dataQueue, numpyImages, numpyGT))
            dataPreparation[proc].daemon = True
            dataPreparation[proc].start()

        self.trainThread(dataQueue, solver)

    def get_class_and_feature_volume(self, net, volume):
        batchsize = self.params['ModelParams']['batchsize']
        h_patch_size = int(self.params['ModelParams']['patchSize'] / 2)

        #meshgrid xx yy zz
        xx = np.arange(h_patch_size + 1,
                       self.params['DataManagerParams']['VolSize'][0] - h_patch_size -1,
                       step=self.params['ModelParams']['SamplingStep'])
        yy = np.arange(h_patch_size + 1,
                       self.params['DataManagerParams']['VolSize'][1] - h_patch_size - 1,
                       step=self.params['ModelParams']['SamplingStep'])
        zz = np.arange(h_patch_size + 1,
                       self.params['DataManagerParams']['VolSize'][2] - h_patch_size - 1,
                       step=self.params['ModelParams']['SamplingStep'])

        xx, yy, zz = np.meshgrid(xx, yy, zz)

        xx = xx.flatten()
        yy = yy.flatten()
        zz = zz.flatten()

        results_label = np.zeros(xx.shape[0], dtype=int)
        results_probability = np.zeros(xx.shape[0], dtype=np.float32)
        results_feature = np.zeros((xx.shape[0], int(self.params['ModelParams']['featLength'])), dtype=np.float32)

        for i in range(int(np.ceil(xx.shape[0] / batchsize))):
            curr_xx = xx[i * batchsize:(i + 1) * batchsize]
            curr_yy = yy[i * batchsize:(i + 1) * batchsize]
            curr_zz = zz[i * batchsize:(i + 1) * batchsize]

            imgPatches = np.zeros((self.params['ModelParams']['batchsize'], 1,
                                   self.params['ModelParams']['patchSize'],
                                   self.params['ModelParams']['patchSize'],
                                   self.params['ModelParams']['patchSize']), dtype=np.float32)

            for x_, y_, z_, k in zip(curr_xx, curr_yy, curr_zz, range(len(curr_xx))):
                imgPatches[k, 0] = volume[x_ - h_patch_size - 1:x_ + h_patch_size,
                                          y_ - h_patch_size - 1:y_ + h_patch_size,
                                          z_ - h_patch_size - 1:z_ + h_patch_size]
            net.blobs['data'].data[...] = imgPatches

            out = net.forward()

            l = np.argmax(out["pred"], axis=1)
            p = np.max(out["pred"], axis=1)
            f = out["fc2_out"]

            results_label[i * batchsize:(i + 1) * batchsize] = l
            results_feature[i * batchsize:(i + 1) * batchsize] = f
            results_probability[i * batchsize:(i + 1) * batchsize] = p

        return results_label.astype(dtype=int), results_probability, results_feature, np.vstack((xx, yy, zz)).T

    def cast_votes_and_segment(self, results_label, results_probability, results_feature, coords, numpyGT):
        votemap = np.zeros((self.params['DataManagerParams']['VolSize'][0],
                  self.params['DataManagerParams']['VolSize'][1],
                  self.params['DataManagerParams']['VolSize'][2]), dtype=np.float32)

        segmentation = np.zeros((self.params['DataManagerParams']['VolSize'][0],
                            self.params['DataManagerParams']['VolSize'][1],
                            self.params['DataManagerParams']['VolSize'][2]), dtype=np.float32)

        denominator = np.zeros((self.params['DataManagerParams']['VolSize'][0],
                                 self.params['DataManagerParams']['VolSize'][1],
                                 self.params['DataManagerParams']['VolSize'][2]), dtype=np.float32)

        results_feature = results_feature[results_label > 0]
        coords = coords[results_label > 0]

        neighbors_idx, votes, seg_patch_coords, seg_patch_vol, distance = self.knn_search(results_feature)

        coords = np.tile(coords, (self.params['ModelParams']['numNeighs'], 1))
        dst_votes = coords + votes

        dst_votes = dst_votes.astype(dtype=int)

        for v, d in zip(dst_votes, distance):
            try:
                if d <= self.params['ModelParams']['maxDist']:
                    votemap[v[0], v[1], v[2]] += 1.0 / (d + 1.0)
            except IndexError:
                pass
        votemap = scipy.ndimage.filters.gaussian_filter(votemap, 3)

        max_loc = np.argmax(votemap)
        xc, yc, zc = np.unravel_index(max_loc, votemap.shape)

        h_seg_patch_size = self.params['ModelParams']['SegPatchRadius']

        reject_votes = np.sqrt(np.sum((dst_votes - np.asarray([xc, yc, zc])) ** 2, 1)) \
                       < self.params['ModelParams']['centrtol']
        w = np.ones_like(distance) / (distance + 1.0)
        w[distance > self.params['ModelParams']['maxDist']] = -1.0

        curr_dst_coords = coords[reject_votes]
        curr_seg_patch_coords = seg_patch_coords[reject_votes]
        curr_seg_patch_vol = seg_patch_vol[reject_votes]
        curr_weight = w[reject_votes]

        patches = self.retrieve_seg_patches(curr_seg_patch_coords, curr_seg_patch_vol, numpyGT)
        #patches has size [n_seg_patches, h, w, d]

        #apply patches in appropriate places

        for p, c, w in zip(patches, curr_dst_coords, curr_weight):
            if w == -1.0:
                continue
            try:
                segmentation[c[0] - h_seg_patch_size[0] - 1:c[0] + h_seg_patch_size[0],
                            c[1] - h_seg_patch_size[1] - 1:c[1] + h_seg_patch_size[1],
                            c[2] - h_seg_patch_size[2] - 1:c[2] + h_seg_patch_size[2]] += p * w

                denominator[c[0] - h_seg_patch_size[0] - 1:c[0] + h_seg_patch_size[0],
                            c[1] - h_seg_patch_size[1] - 1:c[1] + h_seg_patch_size[1],
                            c[2] - h_seg_patch_size[2] - 1:c[2] + h_seg_patch_size[2]] += w
            except ValueError:  # we might want to apply a segpatch where it does not fit. skipping it for now (pass)
                pass

        segmentation /= (denominator+EPS)

        return votemap, segmentation

    def retrieve_seg_patches(self, curr_seg_patch_coords, curr_seg_patch_vol, numpyGT):
        patches = np.zeros((curr_seg_patch_coords.shape[0],
                  self.params['ModelParams']['SegPatchRadius'][0] * 2 + 1,
                  self.params['ModelParams']['SegPatchRadius'][1] * 2 + 1,
                  self.params['ModelParams']['SegPatchRadius'][2] * 2 + 1), dtype=np.float32)
        idx = 0
        for coord, vol in zip(curr_seg_patch_coords, curr_seg_patch_vol):
            try:
                patches[idx] = numpyGT[vol[0]][
                            coord[0] - self.params['ModelParams']['SegPatchRadius'][0] - 1:
                            coord[0] + self.params['ModelParams']['SegPatchRadius'][0],
                            coord[1] - self.params['ModelParams']['SegPatchRadius'][1] - 1:
                            coord[1] + self.params['ModelParams']['SegPatchRadius'][1],
                            coord[2] - self.params['ModelParams']['SegPatchRadius'][2] - 1:
                            coord[2] + self.params['ModelParams']['SegPatchRadius'][2]
                            ]
            except ValueError:  # trying to read form outside the volume
                patches[idx] = np.zeros_like(patches[idx - 1])  # not completely safe
                pass
            idx += 1
        return patches

    def knn_search(self, result_feature):
        distances, indices = self.database.kneighbors(result_feature)
        distances = distances.T
        indices = indices.T

        neighbors_idx = indices.flatten()
        seg_patch_coords = self.coordsDB[neighbors_idx]
        seg_patch_vol = self.volIdxDB[neighbors_idx]
        votes = self.votesDB[neighbors_idx]
        distances = distances.flatten()

        return neighbors_idx, votes, seg_patch_coords, seg_patch_vol, distances

    def create_database(self, net, volumes, annotations):
        self.featDB = np.empty((0, self.params['ModelParams']['featLength']), dtype=np.float32)
        self.coordsDB = np.empty((0, 3), dtype=np.float32)
        self.volIdxDB = np.empty((0, 1), dtype=object)
        self.votesDB = np.empty((0, 3), dtype=np.float32)

        volIdx = 0
        for volumeK, annotationK in zip(volumes, annotations):
            volume = volumes[volumeK]
            annotation = annotations[annotationK]

            _, _, results_feature, coords = \
                self.get_class_and_feature_volume(net, volume)

            centroid = np.reshape(scipy.ndimage.measurements.center_of_mass(annotation), [1, 3])

            valid = annotation[coords[:, 0], coords[:, 1], coords[:, 2]] > 0

            results_feature = results_feature[valid, :]
            coords = coords[valid, :]
            votes = centroid - coords

            self.featDB = np.vstack((self.featDB, results_feature))
            self.coordsDB = np.vstack((self.coordsDB, coords))
            self.volIdxDB = np.vstack((self.volIdxDB,
                                       np.reshape(np.asarray(results_feature.shape[0] * [annotationK], dtype=object),
                                                  (-1, 1))))
            self.votesDB = np.vstack((self.votesDB, votes))

            volIdx += 1
            # create scikit database for NN.
        self.database = NearestNeighbors(n_neighbors=self.params['ModelParams']['numNeighs'],
                                         algorithm='ball_tree').fit(self.featDB)

        with open(self.params['DataManagerParams']['databasePklSavePath'], 'wb') as f:
            pkl.dump((self.database, self.coordsDB, self.volIdxDB, self.featDB, self.votesDB), f)

    def load_database(self):
        with open(self.params['DataManagerParams']['databasePklLoadPath'], 'rb') as f:
            self.database, self.coordsDB, self.volIdxDB, self.featDB, self.votesDB = pkl.load(f)
        if self.params['DataManagerParams']['rebuildDbase']:
            self.database = NearestNeighbors(n_neighbors=self.params['ModelParams']['numNeighs'],
                                            algorithm='brute').fit(self.featDB)

    def test(self):
        self.dataManagerTest = DM.DataManager(self.params['ModelParams']['dirTest'], self.params['ModelParams']['dirResult'], self.params['DataManagerParams'])
        self.dataManagerTest.loadTestData()


        net = caffe.Net(self.params['ModelParams']['prototxtTest'],
                        os.path.join(self.params['ModelParams']['dirSnapshots'],"_iter_" + str(self.params['ModelParams']['snapshot']) + ".caffemodel"),
                        caffe.TEST)

        numpyImages = self.dataManagerTest.getNumpyImages()

        for key in numpyImages:
            mean = np.mean(numpyImages[key][numpyImages[key]>0])
            std = np.std(numpyImages[key][numpyImages[key]>0])
            numpyImages[key] -= mean
            numpyImages[key] /= std


        self.dataManagerTrain = DM.DataManager(self.params['ModelParams']['dirTrain'],
                                               self.params['ModelParams']['dirResult'],
                                               self.params['DataManagerParams'])

        self.dataManagerTrain.loadTrainingData()  # loads in sitk format

        numpyGT = self.dataManagerTrain.getNumpyGT() # GT of the training set, needed in memory to do segmentation

        if self.params['DataManagerParams']['databasePklLoadPath'] is None:
            numpyImagesTrain = self.dataManagerTrain.getNumpyImages()
            for key in numpyImagesTrain:
                mean = np.mean(numpyImagesTrain[key][numpyImagesTrain[key] > 0])
                std = np.std(numpyImagesTrain[key][numpyImagesTrain[key] > 0])
                numpyImagesTrain[key] -= mean
                numpyImagesTrain[key] /= std

            self.create_database(net, numpyImagesTrain, numpyGT)
        else:
            self.load_database()

        results = dict()

        for key in numpyImages:
            results_label, results_probability, results_feature, coords = \
                self.get_class_and_feature_volume(net, numpyImages[key])

            votemap, segmentation = self.cast_votes_and_segment(results_label, results_probability,
                                                                results_feature, coords, numpyGT)

            print(segmentation.shape)
            print('done {}'.format(key))
            results[key] = segmentation

            print("{} foreground voxels".format(np.sum(results[key] > 0)))

            self.dataManagerTest.writeResultsFromNumpyLabel(results[key], key)


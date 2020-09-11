import torch, os, cv2
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from PIL import Image

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(pretrained = False, backbone=cfg.backbone,cls_dim = (cfg.griding_num+1,cls_num_per_lane,4),
                    use_aux=False).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        elif 'aux_' not in k:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=True)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    transform1 = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
#    splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
#    datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]

#    for split, dataset in zip(splits, datasets):
#        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
#        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#        print(split[:-3]+'avi')
#        vout = cv2.VideoWriter(split[:-3]+'avi', fourcc , 30.0, (1640,590))
#        vout = cv2.VideoWriter('1.mp4', fourcc , 30.0, (1640,590))
#        for i, data in enumerate(tqdm.tqdm(loader)):
    cap = cv2.VideoCapture('/home/zty/Lane-detection/demo/2.mp4')
    cv2.namedWindow("outputvideo",cv2.WINDOW_NORMAL)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fout = cv2.VideoWriter('output.avi',fourcc,20.0,(1280,720))
    while True:
        ret, frame = cap.read()
        img = frame
        img1 = frame
        if ret:
            width = frame.shape[1]
            height = frame.shape[0]
            frame = Image.fromarray(frame)
#            frame = frame.cuda()
 #           frame = transform_img({'img': frame})['img']
            frame = img_transforms(frame)
            frame = frame.unsqueeze(0)
#            frame = cv2.resize(frame,(288,800),interpolation=cv2.INTER_CUBIC)
#            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#            img = transforms.ToTensor()(img)
#            img = img.resize(1,3,288,800)
#            img = transform1(img)
            frame = frame.cuda()
            with torch.no_grad():
                out = net(frame)
#            imgs, names = data
#            imgs = imgs.cuda()
#            with torch.no_grad():
#                out = net(imgs)

            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            # import pdb; pdb.set_trace()
#            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            for i in range(out_j.shape[1]):
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * width / 800) - 1, int(height - k * height /590*20) - 1)
                            cv2.circle(img,ppp,5,(0,255,0),-1)
                            
                            cv2.imshow("outputvideo", img)
#                            out.write(img)
#                        cv2.circle(vis,ppp,5,(0,255,0),-1)
#            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#            cv2.imshow('input_video', img1)
#            cv2.imshow('output_video', img)
            fout.write(img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()
#            vout.write(vis)
        
#        vout.release()

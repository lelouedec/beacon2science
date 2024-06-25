from typing import Dict, List, Optional

import torch
from torch import nn,optim
from torch.nn import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import Sequences_dataset
from kornia.geometry.transform import translate
from kornia.enhance.equalization import equalize_clahe
from torch.utils.tensorboard import SummaryWriter
import sys 
import json



#############FUSIOOOOOONNNN################



_NUMBER_OF_COLOR_CHANNELS = 1


def get_channels_at_level(level, filters):
    n_images = 2
    channels = _NUMBER_OF_COLOR_CHANNELS
    flows = 2

    return (sum(filters << i for i in range(level)) + channels + flows) * n_images


class Fusion(nn.Module):
    """The decoder."""

    def __init__(self, n_layers=4, specialized_layers=3, filters=64):
        """
        Args:
            m: specialized levels
        """
        super().__init__()

        # The final convolution that outputs RGB:
        self.output_conv = nn.Conv2d(filters, 1, kernel_size=1)

        # Each item 'convs[i]' will contain the list of convolutions to be applied
        # for pyramid level 'i'.
        self.convs = nn.ModuleList()

        # Create the convolutions. Roughly following the feature extractor, we
        # double the number of filters when the resolution halves, but only up to
        # the specialized_levels, after which we use the same number of filters on
        # all levels.
        #
        # We create the convs in fine-to-coarse order, so that the array index
        # for the convs will correspond to our normal indexing (0=finest level).
        # in_channels: tuple = (128, 202, 256, 522, 512, 1162, 1930, 2442)

        in_channels = get_channels_at_level(n_layers, filters)
        increase = 0
        for i in range(n_layers)[::-1]:
            num_filters = (filters << i) if i < specialized_layers else (filters << specialized_layers)
            convs = nn.ModuleList([
                Conv2d(in_channels, num_filters, size=2, activation=None),
                Conv2d(in_channels + (increase or num_filters), num_filters, size=3),
                Conv2d(num_filters, num_filters, size=3)]
            )
            self.convs.append(convs)
            in_channels = num_filters
            increase = get_channels_at_level(i, filters) - num_filters // 2

    def forward(self, pyramid: List[torch.Tensor]) -> torch.Tensor:
        """Runs the fusion module.

        Args:
          pyramid: The input feature pyramid as list of tensors. Each tensor being
            in (B x H x W x C) format, with finest level tensor first.

        Returns:
          A batch of RGB images.
        Raises:
          ValueError, if len(pyramid) != config.fusion_pyramid_levels as provided in
            the constructor.
        """

        # As a slight difference to a conventional decoder (e.g. U-net), we don't
        # apply any extra convolutions to the coarsest level, but just pass it
        # to finer levels for concatenation. This choice has not been thoroughly
        # evaluated, but is motivated by the educated guess that the fusion part
        # probably does not need large spatial context, because at this point the
        # features are spatially aligned by the preceding warp.
        net = pyramid[-1]

        # Loop starting from the 2nd coarsest level:
        # for i in reversed(range(0, len(pyramid) - 1)):
        for k, layers in enumerate(self.convs):
            i = len(self.convs) - 1 - k
            # Resize the tensor from coarser level to match for concatenation.
            level_size = pyramid[i].shape[2:4]
            net = F.interpolate(net, size=level_size, mode='nearest')
            net = layers[0](net)
            net = torch.cat([pyramid[i], net], dim=1)
            net = layers[1](net)
            net = layers[2](net)
        net = self.output_conv(net)
        return net
    
#############FLOW ESTIMATOOORRR################


class FlowEstimator(nn.Module):
    """Small-receptive field predictor for computing the flow between two images.

    This is used to compute the residual flow fields in PyramidFlowEstimator.

    Note that while the number of 3x3 convolutions & filters to apply is
    configurable, two extra 1x1 convolutions are appended to extract the flow in
    the end.

    Attributes:
      name: The name of the layer
      num_convs: Number of 3x3 convolutions to apply
      num_filters: Number of filters in each 3x3 convolution
    """

    def __init__(self, in_channels: int, num_convs: int, num_filters: int):
        super(FlowEstimator, self).__init__()

        self._convs = nn.ModuleList()
        for i in range(num_convs):
            self._convs.append(Conv2d(in_channels=in_channels, out_channels=num_filters, size=3))
            in_channels = num_filters
        self._convs.append(Conv2d(in_channels, num_filters // 2, size=1))
        in_channels = num_filters // 2
        # For the final convolution, we want no activation at all to predict the
        # optical flow vector values. We have done extensive testing on explicitly
        # bounding these values using sigmoid, but it turned out that having no
        # activation gives better results.
        self._convs.append(Conv2d(in_channels, 2, size=1, activation=None))

    def forward(self, features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
        """Estimates optical flow between two images.

        Args:
          features_a: per pixel feature vectors for image A (B x H x W x C)
          features_b: per pixel feature vectors for image B (B x H x W x C)

        Returns:
          A tensor with optical flow from A to B
        """
        net = torch.cat([features_a, features_b], dim=1)
        for conv in self._convs:
            net = conv(net)
        return net


class PyramidFlowEstimator(nn.Module):
    """Predicts optical flow by coarse-to-fine refinement.
    """

    def __init__(self, filters: int = 64,
                 flow_convs: tuple = (3, 3, 3, 3),
                 flow_filters: tuple = (32, 64, 128, 256)):
        super(PyramidFlowEstimator, self).__init__()

        in_channels = filters << 1
        print("IN CHANNELS   ",in_channels)
        predictors = []
        for i in range(len(flow_convs)):
            predictors.append(
                FlowEstimator(
                    in_channels=in_channels,
                    num_convs=flow_convs[i],
                    num_filters=flow_filters[i]))
            in_channels += filters << (i + 2)
        self._predictor = predictors[-1]
        self._predictors = nn.ModuleList(predictors[:-1][::-1])

    def forward(self, feature_pyramid_a: List[torch.Tensor],
                feature_pyramid_b: List[torch.Tensor]) -> List[torch.Tensor]:
        """Estimates residual flow pyramids between two image pyramids.

        Each image pyramid is represented as a list of tensors in fine-to-coarse
        order. Each individual image is represented as a tensor where each pixel is
        a vector of image features.

        flow_pyramid_synthesis can be used to convert the residual flow
        pyramid returned by this method into a flow pyramid, where each level
        encodes the flow instead of a residual correction.

        Args:
          feature_pyramid_a: image pyramid as a list in fine-to-coarse order
          feature_pyramid_b: image pyramid as a list in fine-to-coarse order

        Returns:
          List of flow tensors, in fine-to-coarse order, each level encoding the
          difference against the bilinearly upsampled version from the coarser
          level. The coarsest flow tensor, e.g. the last element in the array is the
          'DC-term', e.g. not a residual (alternatively you can think of it being a
          residual against zero).
        """
        levels = len(feature_pyramid_a)
        v = self._predictor(feature_pyramid_a[-1], feature_pyramid_b[-1])
        residuals = [v]
        for i in range(levels - 2, len(self._predictors) - 1, -1):
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = self._predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v

        for k, predictor in enumerate(self._predictors):
            i = len(self._predictors) - 1 - k
            # Upsamples the flow to match the current pyramid level. Also, scales the
            # magnitude by two to reflect the new size.
            level_size = feature_pyramid_a[i].shape[2:4]
            v = F.interpolate(2 * v, size=level_size, mode='bilinear')
            # Warp feature_pyramid_b[i] image based on the current flow estimate.
            warped = warp(feature_pyramid_b[i], v)
            # Estimate the residual flow between pyramid_a[i] and warped image:
            v_residual = predictor(feature_pyramid_a[i], warped)
            residuals.insert(0, v_residual)
            v = v_residual + v
        return residuals
    


def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region


def load_image(path, align=64):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region


def build_image_pyramid(image: torch.Tensor, pyramid_levels: int = 3) -> List[torch.Tensor]:
    """Builds an image pyramid from a given image.

    The original image is included in the pyramid and the rest are generated by
    successively halving the resolution.

    Args:
      image: the input image.
      options: film_net options object

    Returns:
      A list of images starting from the finest with options.pyramid_levels items
    """

    pyramid = []
    for i in range(pyramid_levels):
        pyramid.append(image)
        if i < pyramid_levels - 1:
            image = F.avg_pool2d(image, 2, 2)
    return pyramid


def warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Backward warps the image using the given flow.

    Specifically, the output pixel in batch b, at position x, y will be computed
    as follows:
      (flowed_y, flowed_x) = (y+flow[b, y, x, 1], x+flow[b, y, x, 0])
      output[b, y, x] = bilinear_lookup(image, b, flowed_y, flowed_x)

    Note that the flow vectors are expected as [x, y], e.g. x in position 0 and
    y in position 1.

    Args:
      image: An image with shape BxHxWxC.
      flow: A flow with shape BxHxWx2, with the two channels denoting the relative
        offset in order: (dx, dy).
    Returns:
      A warped image.
    """
    flow = -flow.flip(1)

    dtype = flow.dtype
    device = flow.device

    # warped = tfa_image.dense_image_warp(image, flow)
    # Same as above but with pytorch
    ls1 = 1 - 1 / flow.shape[3]
    ls2 = 1 - 1 / flow.shape[2]

    normalized_flow2 = flow.permute(0, 2, 3, 1) / torch.tensor(
        [flow.shape[2] * .5, flow.shape[3] * .5], dtype=dtype, device=device)[None, None, None]
    normalized_flow2 = torch.stack([
        torch.linspace(-ls1, ls1, flow.shape[3], dtype=dtype, device=device)[None, None, :] - normalized_flow2[..., 1],
        torch.linspace(-ls2, ls2, flow.shape[2], dtype=dtype, device=device)[None, :, None] - normalized_flow2[..., 0],
    ], dim=3)

    warped = F.grid_sample(image, normalized_flow2,
                           mode='bilinear', padding_mode='border', align_corners=False)
    return warped.reshape(image.shape)


def multiply_pyramid(pyramid: List[torch.Tensor],
                     scalar: torch.Tensor) -> List[torch.Tensor]:
    """Multiplies all image batches in the pyramid by a batch of scalars.

    Args:
      pyramid: Pyramid of image batches.
      scalar: Batch of scalars.

    Returns:
      An image pyramid with all images multiplied by the scalar.
    """
    # To multiply each image with its corresponding scalar, we first transpose
    # the batch of images from BxHxWxC-format to CxHxWxB. This can then be
    # multiplied with a batch of scalars, then we transpose back to the standard
    # BxHxWxC form.
    return [image * scalar[..., None, None] for image in pyramid]


def flow_pyramid_synthesis(
        residual_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Converts a residual flow pyramid into a flow pyramid."""
    flow = residual_pyramid[-1]
    flow_pyramid: List[torch.Tensor] = [flow]
    for residual_flow in residual_pyramid[:-1][::-1]:
        level_size = residual_flow.shape[2:4]
        flow = F.interpolate(2 * flow, size=level_size, mode='bilinear')
        flow = residual_flow + flow
        flow_pyramid.insert(0, flow)
    return flow_pyramid


def pyramid_warp(feature_pyramid: List[torch.Tensor],
                 flow_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Warps the feature pyramid using the flow pyramid.

    Args:
      feature_pyramid: feature pyramid starting from the finest level.
      flow_pyramid: flow fields, starting from the finest level.

    Returns:
      Reverse warped feature pyramid.
    """
    warped_feature_pyramid = []
    for features, flow in zip(feature_pyramid, flow_pyramid):
        warped_feature_pyramid.append(warp(features, flow))
    return warped_feature_pyramid


def concatenate_pyramids(pyramid1: List[torch.Tensor],
                         pyramid2: List[torch.Tensor]) -> List[torch.Tensor]:
    """Concatenates each pyramid level together in the channel dimension."""
    result = []
    for features1, features2 in zip(pyramid1, pyramid2):
        result.append(torch.cat([features1, features2], dim=1))
    return result


class Conv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, size, activation: Optional[str] = 'relu'):
        assert activation in (None, 'relu')
        super().__init__(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=size,
                padding='same' if size % 2 else 0)
        )
        self.size = size
        self.activation = nn.LeakyReLU(.2) if activation == 'relu' else None

    def forward(self, x):
        if not self.size % 2:
            x = F.pad(x, (0, 1, 0, 1))
        y = self[0](x)
        if self.activation is not None:
            y = self.activation(y)
        return y
    


class SubTreeExtractor(nn.Module):
    """Extracts a hierarchical set of features from an image.

    This is a conventional, hierarchical image feature extractor, that extracts
    [k, k*2, k*4... ] filters for the image pyramid where k=options.sub_levels.
    Each level is followed by average pooling.
    """

    def __init__(self, in_channels=3, channels=64, n_layers=4):
        super().__init__()
        convs = []
        for i in range(n_layers):
            convs.append(nn.Sequential(
                Conv2d(in_channels, (channels << i), 3),
                Conv2d((channels << i), (channels << i), 3)
            ))
            in_channels = channels << i
        self.convs = nn.ModuleList(convs)

    def forward(self, image: torch.Tensor, n: int) -> List[torch.Tensor]:
        """Extracts a pyramid of features from the image.

        Args:
          image: TORCH.Tensor with shape BATCH_SIZE x HEIGHT x WIDTH x CHANNELS.
          n: number of pyramid levels to extract. This can be less or equal to
           options.sub_levels given in the __init__.
        Returns:
          The pyramid of features, starting from the finest level. Each element
          contains the output after the last convolution on the corresponding
          pyramid level.
        """
        head = image
        pyramid = []
        for i, layer in enumerate(self.convs):
            head = layer(head)
            pyramid.append(head)
            if i < n - 1:
                head = F.avg_pool2d(head, kernel_size=2, stride=2)
        return pyramid


class FeatureExtractor(nn.Module):
    """Extracts features from an image pyramid using a cascaded architecture.
    """

    def __init__(self, in_channels=3, channels=64, sub_levels=4):
        super().__init__()
        self.extract_sublevels = SubTreeExtractor(in_channels, channels, sub_levels)
        self.sub_levels = sub_levels

    def forward(self, image_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
        """Extracts a cascaded feature pyramid.

        Args:
          image_pyramid: Image pyramid as a list, starting from the finest level.
        Returns:
          A pyramid of cascaded features.
        """
        sub_pyramids: List[List[torch.Tensor]] = []
        for i in range(len(image_pyramid)):
            # At each level of the image pyramid, creates a sub_pyramid of features
            # with 'sub_levels' pyramid levels, re-using the same SubTreeExtractor.
            # We use the same instance since we want to share the weights.
            #
            # However, we cap the depth of the sub_pyramid so we don't create features
            # that are beyond the coarsest level of the cascaded feature pyramid we
            # want to generate.
            capped_sub_levels = min(len(image_pyramid) - i, self.sub_levels)
            sub_pyramids.append(self.extract_sublevels(image_pyramid[i], capped_sub_levels))
        # Below we generate the cascades of features on each level of the feature
        # pyramid. Assuming sub_levels=3, The layout of the features will be
        # as shown in the example on file documentation above.
        feature_pyramid: List[torch.Tensor] = []
        for i in range(len(image_pyramid)):
            features = sub_pyramids[i][0]
            for j in range(1, self.sub_levels):
                if j <= i:
                    features = torch.cat([features, sub_pyramids[i - j][j]], dim=1)
            feature_pyramid.append(features)
        return feature_pyramid


class Interpolator(nn.Module):
    def __init__(
            self,
            pyramid_levels=7,
            fusion_pyramid_levels=5,
            specialized_levels=3,
            sub_levels=4,
            filters=64,
            flow_convs=(3, 3, 3, 3),
            flow_filters=(32, 64, 128, 256),
    ):
        super().__init__()
        self.pyramid_levels = pyramid_levels
        self.fusion_pyramid_levels = fusion_pyramid_levels

        self.extract = FeatureExtractor(1, filters, sub_levels)
        self.predict_flow = PyramidFlowEstimator(filters, flow_convs, flow_filters)
        self.fuse = Fusion(sub_levels, specialized_levels, filters)

    def shuffle_images(self, x0, x1):
        return [
            build_image_pyramid(x0, self.pyramid_levels),
            build_image_pyramid(x1, self.pyramid_levels)
        ]

    def debug_forward(self, x0, x1, batch_dt) -> Dict[str, List[torch.Tensor]]:
        image_pyramids = self.shuffle_images(x0, x1)

        # Siamese feature pyramids:
        feature_pyramids = [self.extract(image_pyramids[0]), self.extract(image_pyramids[1])]

        # Predict forward flow.
        forward_residual_flow_pyramid = self.predict_flow(feature_pyramids[0], feature_pyramids[1])

        # Predict backward flow.
        backward_residual_flow_pyramid = self.predict_flow(feature_pyramids[1], feature_pyramids[0])

        # Concatenate features and images:

        # Note that we keep up to 'fusion_pyramid_levels' levels as only those
        # are used by the fusion module.

        forward_flow_pyramid = flow_pyramid_synthesis(forward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        backward_flow_pyramid = flow_pyramid_synthesis(backward_residual_flow_pyramid)[:self.fusion_pyramid_levels]

        # We multiply the flows with t and 1-t to warp to the desired fractional time.
        #
        # Note: In film_net we fix time to be 0.5, and recursively invoke the interpo-
        # lator for multi-frame interpolation. Below, we create a constant tensor of
        # shape [B]. We use the `time` tensor to infer the batch size.
        backward_flow = multiply_pyramid(backward_flow_pyramid, batch_dt)
        forward_flow = multiply_pyramid(forward_flow_pyramid, 1 - batch_dt)

        pyramids_to_warp = [
            concatenate_pyramids(image_pyramids[0][:self.fusion_pyramid_levels],
                                      feature_pyramids[0][:self.fusion_pyramid_levels]),
            concatenate_pyramids(image_pyramids[1][:self.fusion_pyramid_levels],
                                      feature_pyramids[1][:self.fusion_pyramid_levels])
        ]

        # Warp features and images using the flow. Note that we use backward warping
        # and backward flow is used to read from image 0 and forward flow from
        # image 1.
        forward_warped_pyramid = pyramid_warp(pyramids_to_warp[0], backward_flow)
        backward_warped_pyramid = pyramid_warp(pyramids_to_warp[1], forward_flow)

        aligned_pyramid = concatenate_pyramids(forward_warped_pyramid,
                                                    backward_warped_pyramid)
        aligned_pyramid = concatenate_pyramids(aligned_pyramid, backward_flow)
        aligned_pyramid = concatenate_pyramids(aligned_pyramid, forward_flow)

        return {
            'image': [self.fuse(aligned_pyramid)],
            'forward_residual_flow_pyramid': forward_residual_flow_pyramid,
            'backward_residual_flow_pyramid': backward_residual_flow_pyramid,
            'forward_flow_pyramid': forward_flow_pyramid,
            'backward_flow_pyramid': backward_flow_pyramid,
        }

    def forward(self, x0, x1, batch_dt) -> torch.Tensor:
        return self.debug_forward(x0, x1, batch_dt)['image'][0]
    

if __name__ == "__main__":
    with open(sys.argv[1]) as handle:
        config = json.load(handle)



    device = torch.device("cpu")
    if(torch.backends.mps.is_available()):
        device = torch.device("mps")
    elif(torch.cuda.is_available()):
        device = torch.device("cuda")

    precision = torch.float32


    model = Interpolator()
    if(torch.cuda.device_count() >1):
        model = torch.nn.DataParallel(model)

    model.to(device=device, dtype=precision)

    # dataset = Sequences_dataset.FinalDatasetSequences(512,"/gpfs/data/fs72241/lelouedecj/",training=True,validation=False)
    # dataset_validation = Sequences_dataset.FinalDatasetSequences(512,"/gpfs/data/fs72241/lelouedecj/",training=False,validation=True)


    dataset = Sequences_dataset.FinalDatasetSequences(512,"../",training=True,validation=False)
    dataset_validation = Sequences_dataset.FinalDatasetSequences(512,"../",training=False,validation=True)


    # dataset = Sequences_dataset.FinalDatasetSequences(512,"/Volumes/Data_drive/",training=True,validation=False)
    # dataset_validation = Sequences_dataset.FinalDatasetSequences(512,"/Volumes/Data_drive/",training=False,validation=True)
 
    minibacth = config["minibatch"]
    dataloader = torch.utils.data.DataLoader(
                                                dataset,
                                                batch_size=minibacth,
                                                shuffle=False,
                                                num_workers=int(minibacth/2),
                                                pin_memory=False
                                            )
    
    dataloader_validation = torch.utils.data.DataLoader(
                                                dataset_validation,
                                                batch_size=minibacth,
                                                shuffle=True,
                                                num_workers=int(minibacth/2),
                                                pin_memory=False
                                            )
    
    pixel_looser = nn.L1Loss(reduction="mean")
    optimizer = optim.Adam(model.parameters(),1e-5)
    writer = SummaryWriter()
    best_validation = 20.0


    losses1 = []
    losses2 = []
    losses1_v = []
    losses2_v = []
    for i in range(0,400):
        l1s = []
        l2s = []
        l1sv = []
        l2sv = []

        for data in dataloader:
            S1 = data["IM1"].to(device)
            S2 = data["IM2"].to(device)
            S3 = data["IM3"].to(device)
            S4 = data["IM4"].to(device)
            


            dt1 = torch.ones((S1.shape[0],1)).to(device) * data["ratio1"].float().to(device)
            dt2 = torch.ones((S1.shape[0],1)).to(device) * data["ratio2"].float().to(device)

            print(dt1.shape)

            

            optimizer.zero_grad()
            output25 = model(S1,S2,dt1)
            loss1 = pixel_looser(output25,S3)
            # if(config["diffloss"]):
            #     loss1+=0.1
            loss1.backward()
            optimizer.step()


            optimizer.zero_grad()
            output75 = model(S1,S2,dt2)
            loss2 = pixel_looser(output75,S4)
            loss2.backward()
            optimizer.step()

            l1s.append(loss1.item())
            l2s.append(loss2.item())
        losses1.append(np.array(l1s).mean())
        losses2.append(np.array(l2s).mean())
        
        with torch.no_grad():
            for data in dataloader_validation:


                S1 = data["IM1"].to(device)
                S2 = data["IM2"].to(device)
                S3 = data["IM3"].to(device)
                S4 = data["IM4"].to(device)
                


                dt1 = torch.ones((S1.shape[0],1)).to(device) * data["ratio1"].float().to(device)
                dt2 = torch.ones((S1.shape[0],1)).to(device) * data["ratio2"].float().to(device)

                

                output25 = model(S1,S2,dt1)
                loss1_v = pixel_looser(output25,S3)


                output75 = model(S1,S2,dt2)
                loss2_v = pixel_looser(output75,S4)



            
                l1sv.append(loss1_v.item())
                l2sv.append(loss2_v.item())


            losses1_v.append(np.array(l1sv).mean())
            losses2_v.append(np.array(l2sv).mean())


        writer.add_scalar('train 1', np.array(losses1)[-1], i)
        writer.add_scalar('train 2', np.array(losses2)[-1], i)
        writer.add_scalar('val 1', np.array(losses1_v)[-1], i)
        writer.add_scalar('val 2', np.array(losses2_v)[-1], i)
        
        writer.add_image('s1', S1[0,0,:,:], i, dataformats='HW')
        writer.add_image('s2', S2[0,0,:,:], i, dataformats='HW')
        writer.add_image('s3', output25[0,0,:,:], i, dataformats='HW')
        writer.add_image('s4', output75[0,0,:,:], i, dataformats='HW')

        writer.add_image('gts3', S3[0,0,:,:], i, dataformats='HW')
        writer.add_image('gts4', S4[0,0,:,:], i, dataformats='HW')
        


        if(len(losses1_v)>1):
            if(losses1_v[-1]<best_validation):
                torch.save(model.module.state_dict(), "FILM_model1.pth")
                best_validation = losses1_v[-1]
        else:
            torch.save(model.module.state_dict(), "FILM_model1.pth")
    

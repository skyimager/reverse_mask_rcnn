from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from numba import jit

import tensorflow as tf
import buzzard as buzz
import numpy as np

def predict_image_batch(batch, model):
    out = model.predict(batch)
    return out

def segmentation_inference_batch_generator(tile_index_batch, tiles, ds):
    batch = []
    for indexes in tile_index_batch:
        i = indexes[0]
        j = indexes[1]
        rgb_array = ds.rgb.get_data(channels=[0,1,2], fp=tiles[i,j], dst_nodata=0)
        batch.append(rgb_array)
    batch = np.array(batch)
    return batch

def split_tiles_in_n_batch(tile_index_list, batch_size=4):
    batch_list = [tile_index_list[i*batch_size:(i+1)*batch_size] for i in range((len(tile_index_list)+batch_size-1)//batch_size)]
    return batch_list

def get_aoi_tile_indexes(tiles):
    tile_index_list = [(i,y) for i in range(tiles.shape[0]) for y in range(tiles.shape[1])]
    return tile_index_list

def tile_image(tile_size, overlapx, overlapy, fp):

    tiles = fp.tile((tile_size,tile_size), overlapx=overlapx, overlapy=overlapy, boundary_effect='extend', boundary_effect_locus='br')
    orgH = tile_size + (tiles.shape[0]-1)*(tile_size-overlapx)
    orgW = tile_size + (tiles.shape[1]-1)*(tile_size-overlapy)

    return tiles, orgH, orgW 

@jit(parallel=True, forceobj=True)
def post_process_predicted_probability_map_batch(tile_index_batch, predicted_batch, 
            orgH, orgW, tile_size, prv_mean, overlapx, overlapy):
    
    all_results = []
    for index, predicted in enumerate(predicted_batch):
        # predicted = predicted*weight_matrix
        ts = [tile_size, tile_size]
        i = tile_index_batch[index][0]
        j = tile_index_batch[index][1]
        temp = np.zeros((orgH, orgW, 5), dtype=np.float32)
        start_row = int(i*tile_size-i*overlapx)
        end_row = start_row + tile_size
        start_col = int(j*tile_size-j*overlapy)
        end_col = start_col + tile_size

        temp[start_row:end_row, start_col:end_col] = predicted
        prv_mean = (prv_mean + temp)

    return prv_mean

def untile_and_predict_batch(tiles, model, ds, tile_size, orgH, orgW, overlapx, overlapy, batch_size):
    aoi_tile_index_list = get_aoi_tile_indexes(tiles)
    tile_index_batch_list = split_tiles_in_n_batch(aoi_tile_index_list, batch_size=batch_size)

    fp_mean = np.zeros((orgH,orgW, 5), dtype=np.float32)
    with tqdm(total=len(tile_index_batch_list)) as t:
        for index, tile_index_batch in enumerate(tile_index_batch_list):
            inference_batch = segmentation_inference_batch_generator(tile_index_batch, tiles, ds)
            predicted_batch = predict_image_batch(inference_batch, model)
            fp_mean = post_process_predicted_probability_map_batch(tile_index_batch, predicted_batch, orgH, orgW, 
                                                        tile_size, fp_mean, overlapx, overlapy)
            t.update()

    return fp_mean

def fix_probabilitymap(predicted_probamap, fp, tile_size, overlapx, overlapy):
    # tilling padding
    y = int(tile_size-overlapy)
    x = int(tile_size-overlapx)
    
    #fixing the central block
    predicted_probamap[y:-y,x:-x,:] = predicted_probamap[y:-y,x:-x,:]/4
    
    #fixing 4 border strips
    predicted_probamap[0:y,x:-x,:] = predicted_probamap[0:y,x:-x,:]/2
    predicted_probamap[-y:,x:-x,:] = predicted_probamap[-y:,x:-x,:]/2
    predicted_probamap[y:-y,0:x,:] = predicted_probamap[y:-y,0:x,:]/2
    predicted_probamap[y:-y,-x:,:] = predicted_probamap[y:-y,-x:,:]/2
    
    #rounding values to fix edge cases
    predicted_probamap = np.around(predicted_probamap, decimals=2, out=None)

    #size check
    predicted_probamap = predicted_probamap[:fp.rsize[1],:fp.rsize[0]]

    return predicted_probamap

def predict_map(model, ds_rgb, fp, tile_size, overlapx, overlapy, batch_size):

    tiles, orgH, orgW = tile_image(tile_size, overlapx, overlapy, fp)
    fp_mean = untile_and_predict_batch(tiles, model, ds_rgb, tile_size, orgH, orgW, overlapx, overlapy, batch_size)
    fp_mean = fix_probabilitymap(fp_mean.copy(), fp, tile_size, overlapx, overlapy)

    return fp_mean

def predict_from_file(rgb_path, model, downsampling_factor=1, tile_size=256, no_of_gpu=1, batch_size=2):

    ds_rgb = buzz.Dataset(allow_interpolation=True)
    ds_rgb.open_raster('rgb', rgb_path)

    fp= buzz.Footprint(
            tl=ds_rgb.rgb.fp.tl,
            size=ds_rgb.rgb.fp.size,
            rsize=ds_rgb.rgb.fp.rsize/downsampling_factor,
    ) #unsampling

    overlapx = int(tile_size/2)
    overlapy = int(tile_size/2)

    try:
        predicted_probamap = predict_map(model, ds_rgb, fp, tile_size, overlapx, overlapy, batch_size)
    except Exception as e:
        print(e)
        print("Retrying prediction with reduced batch size")
        predicted_probamap = predict_map(model, ds_rgb, fp, tile_size, overlapx, overlapy, batch_size//2)

    return predicted_probamap, fp
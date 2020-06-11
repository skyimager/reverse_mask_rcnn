# reverse_mask_rcnn
Building detection in dense and cluttered regions using Reverse Mask R-CNN

## Prerequsities
- You would need to install Docker, a standard OS virtualization software that allows for easy and scalable deployment. You should follow [the instructions here](https://docs.docker.com/install/) to set up for your machine. 
- You would also need to install [prerequisites to run NVIDIA docker](https://github.com/NVIDIA/nvidia-docker) as we are using GPU for inference.
- After this, you should clone this repository:
```
git clone https://github.com/skyimager/reverse_mask_rcnn.git
cd reverse_mask_rcnn
```

## Evaluation
We can setup a server for a dockerised inference of the model using the below Makefile targets :

```
make docker_build
make docker_run
```

To run a live evaluation, w can submit a request:
```
curl -X POST http://localhost:5000/api/predict -F image_file=@$(pwd)/images/chicogo.tif
```

Which will sends the test image onto the server and dump the corresponding predicted probability map into the `results` folder of the project


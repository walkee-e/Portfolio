### How to run the code

1. This is a dockerised project. So, for seamless execution. Create a docker image using the given docker file
2. `docker build -t your-image-name . `
 paste this command and replace “your-image-name” with the desired name. The full stop at the end is part of the command.
3. `docker run -it --rm my-image bash`
  Now your docker container is ready and running
4. `Cd root`
5. Git clone the repo again, this time in root directory of the container.
6. Run the following commands:
`conda install compilers`
`conda install -c conda-forge gcc`
`conda install bottleneck`
`pip install -r requirements.txt`
7. For the LunarLander environment, you can __run the main.py__ in with Python. The model name can chosen in "LIF", "HH", "LIF_HH", "HH_MODIFIED", "HH_CAN" and "IZH". . This file trains each model and returns the rewards in the deep reinforcement learning experiment. 
````
python main.py --model_name LIF
````
8. You can run the performance comparison code to generate the graphs using
`python comparison.py`
9. You can run the model comparison code to generate graphs using
`python graph.py`

### Output

The image outputs of the comparison can be found in the graphs folder



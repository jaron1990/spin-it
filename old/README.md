# Make It Stand: The RUJUM Addition

This is an extended implementation of the (["Make It Stand: Balancing Shapes for 3D Fabrication"](https://igl.ethz.ch/projects/make-it-stand/make-it-stand-siggraph-2013-prevost-et-al.pdf)) paper from 2013.<br>
This project extend the proposed method described in “make it stand” in order to print artistic ״rujums״.<br>

A rujum, or "cairn" in English, is a man-made pile (or stack) of stones, that is often challenging to balance.<br>

![](rep_images/rujum.png?raw=true)

<br>

## Installation

1.  Clone the repo:
```bash
git clone https://github.com/yael-vinker/make_it_stand_rujum.git
cd make_it_stand_rujum
```
2. Create a new environment and install the libraries:
```bash
python3.6 -m venv rujum_venv
source rujum_venv/bin/activate
pip install -r requirements.txt
```
<br>

## Run Demo

The input stones to balance should be located under "resources" directory, sorted from bottom to top.<br>
For example, these are the stones we have used:<br>
![](rep_images/stones.png?raw=true)

<br>
<br>

To balance the stones, from "make_it_stand_rujum" run:
```bash
python run_balance.py
```
The resulting stl files will be saved to the "results" folder.<br>
You should see a visualisation of the balanced rujum:<br>
Where the pink areas are the filled voxels, in green are the safety regions.
<br>
![](rep_images/balanced_ruj.png?raw=true)
<br>
<br>
And our balanced printed version:
<br>
![](rep_images/printed.png?raw=true)

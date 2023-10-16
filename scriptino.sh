#!/bin/bash

src_dir="/home/gabriele/Desktop/provavideo/"
dest_dir="/home/gabriele/Desktop/provavideo2"
num_images=30
shuf -zn "$num_images" -e "$src_dir"*.jpg | xargs -0 mv -t "$dest_dir"


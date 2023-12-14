import numpy as np
from matplotlib import pyplot as plt
import cv2
import os, shutil
import json
from math import pi, acos, cos, sin, sqrt
from bs4 import BeautifulSoup

temp_path = ".\\temp\\"

def parse_param(filename: str):
    with open(filename, "r") as f:
        param = json.load(f)
        return np.array([param["height"], param["width"]], dtype=np.int32), param["linearGradients"]

def parse_region_name(key: str): return tuple(map(int, key.strip()[1:].split("-")))

def parse_bounding_box(mask: np.ndarray)->np.ndarray:
    pixels = np.argwhere(mask==0)[:, :2]
    p_min = pixels.min(0) - 1
    p_max = pixels.max(0) + 1
    return p_min, p_max

def parse_theta(parameter)->float:
    parameter_array = np.array(parameter)
    if np.all(np.abs(parameter_array[:, 0]) < 1e-7): return pi / 2
    elif np.all(np.abs(parameter_array[:, 1]) < 1e-7): return 0
    else:
        axis = np.where(parameter_array[:, 0] != 0)[0][0]
        g_x, g_y = parameter_array[axis, 0], parameter_array[axis, 1]
        if g_y < 0: g_x = -g_x
        return acos(g_x / sqrt(g_x ** 2 + g_y ** 2))

def parse_gradient_tag(p_min, p_max, theta, parameter, layer_index, region_index)->str:
    parameter_array = np.array(parameter)
    start, end = np.array(p_min), np.array(p_max)
    normal = np.array([cos(theta), sin(theta)], dtype=np.float32)
    if theta >= pi / 2:
        start[0], end[0] = end[0], start[0]
    diagonal = end - start
    end = start + np.dot(diagonal, normal) * normal
    start_color = np.matmul(parameter_array, np.concatenate([start, [1]]))
    end_color = np.matmul(parameter_array, np.concatenate([end, [1]]))
    start_color = np.maximum(0, np.minimum(start_color, 1))
    end_color = np.maximum(0, np.minimum(end_color, 1))
    start_rgb = np.floor(start_color[:3] * 255).astype(np.int32)
    end_rgb = np.floor(end_color[:3] * 255).astype(np.int32)
    start_alpha, end_alpha = start_color[3], end_color[3]
    x1 = (start[1] - p_min[1]) / (p_max[1] - p_min[1])
    y1 = (p_max[0] - start[0]) / (p_max[0] - p_min[0])
    x2 = (end[1] - p_min[1]) / (p_max[1] - p_min[1])
    y2 = (p_max[0] - end[0]) / (p_max[0] - p_min[0])
    return (f'<linearGradient id="linear-gradient-{layer_index}-{region_index}" x1="{100 * x1:.2f}%" y1="{100 * y1:.2f}%" x2="{100 * x2:.2f}%" y2="{100 * y2:.2f}%">'
        f'<stop offset="0%" style="stop-color:rgb({start_rgb[0]},{start_rgb[1]},{start_rgb[2]});stop-opacity:{start_alpha:.3f}" />'
        f'<stop offset="100%" style="stop-color:rgb({end_rgb[0]},{end_rgb[1]},{end_rgb[2]});stop-opacity:{end_alpha:.3f}" />'
        '</linearGradient>')

def parse_path_tag(mask: np.ndarray, layer_index, region_index)->str:
    plt.imsave(temp_path + f"{layer_index}-{region_index}.bmp", mask)
    os.system(f".\\potrace-1.16.win64\\potrace.exe -o {temp_path}{layer_index}-{region_index}.svg -s {temp_path}{layer_index}-{region_index}.bmp")
    with open(temp_path + f"{layer_index}-{region_index}.svg", "r") as f:
        soup = BeautifulSoup(f.read(), features="xml")
        group_tag = soup.find(name="g")
        path_tag = soup.find(name="path")
        path_tag["transform"] = group_tag["transform"]
        path_tag["fill"] = f"url(#linear-gradient-{layer_index}-{region_index})"
        path_tag["id"] = f"region-{layer_index}-{region_index}"
        return str(path_tag)

def generate_svg_content(gradient_tags, group_tags, shape: np.ndarray)->str:
    svg_header = ('<?xml version="1.0" standalone="no"?>'
        '<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 20010904//EN" "http://www.w3.org/TR/2001/REC-SVG-20010904/DTD/svg10.dtd">'
        f'<svg version="1.0" xmlns="http://www.w3.org/2000/svg" width="{shape[1]}pt" height="{shape[0]}pt" viewBox="0 0 {shape[1]} {shape[0]}" preserveAspectRatio="xMidYMid meet">')
    svg_footer = '</svg>'
    defs = "<defs>" + "".join(gradient_tags) + "</defs>"
    geometry = "".join(group_tags)
    return svg_header + defs + geometry + svg_footer

dirc = "../Data"
case = "1-Syn1"

if __name__ == "__main__":
    #source_path = input('Please input the directory containing "param.json" and masks: ')
    source_path = dirc + "/" + case + "/results/for_vectorize/0/"
    if not source_path.endswith("\\"): source_path += "\\"
    shape, linear_gradients = parse_param(source_path + "param.json")
    gradient_tags = []
    group_tags = []
    if os.path.exists(temp_path): shutil.rmtree(temp_path)
    os.mkdir(temp_path)
    for layer_parameters in linear_gradients:
        path_tags = []
        for region_name, parameter in layer_parameters.items():
            layer_index, region_index = parse_region_name(region_name)
            mask = cv2.imread(source_path + f"mask_{layer_index}_{region_index}.png")
            mask = cv2.bitwise_not(mask)
            p_min, p_max = parse_bounding_box(mask)
            p_min = p_min / shape
            p_max = p_max / shape
            theta = parse_theta(parameter)
            gradient_tag = parse_gradient_tag(p_min, p_max, theta, parameter, layer_index, region_index)
            gradient_tags.append(gradient_tag)
            path_tag = parse_path_tag(mask, layer_index, region_index)
            path_tags.append(path_tag)
        group_tags.append(f'<g id="layer-{layer_index}">' + "".join(path_tags) + '</g>')

    with open("./svgs/" + case + ".svg", "w") as f:
        content = generate_svg_content(gradient_tags, group_tags, shape)
        f.write(content)
    if os.path.exists(temp_path): shutil.rmtree(temp_path)
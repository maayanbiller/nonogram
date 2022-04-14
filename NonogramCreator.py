import pygame.gfxdraw
import numpy as np
import sys
from PIL import Image
import skimage.color
import skimage.filters
from tkinter import filedialog
import logging
from itertools import permutations
from math import *

from pygame.event import Event

pygame.init()

run = True
win_color = (0, 0, 0)
is_debug = sys.gettrace() != None
win_width = 1920
win_height = 1080
win = pygame.display.set_mode((win_width, win_height), pygame.FULLSCREEN)
if is_debug:
    win_width -= 300
    win_height -= 300
    win = pygame.display.set_mode((win_width, win_height))

# ngram_puzzle = np.random.random((10,20))
# ngram_puzzle = ngram_puzzle > 0.6

def draw_text(text, pos, font, color, disp_update=True):
    txt_surface = font.render(text, True, color)
    win.blit(txt_surface, pos)
    if disp_update:
        pygame.display.update()


def convert_img_file_to_bool_arr(file_name, pixelization_coef= 1):
    image = skimage.color.rgb2gray(skimage.io.imread(file_name))
    otsu_threshold = skimage.filters.threshold_otsu(image)
    return (image > (otsu_threshold + threshold_addition))[::pixelization_coef,::pixelization_coef] * 1


def get_row_sequences(row):
    seq_count = [0]
    for val in row:
        if val:
            seq_count[-1] += 1
        elif seq_count[-1] != 0:
            seq_count.append(0)
    if seq_count[-1] == 0:
        seq_count.remove(0)
    return np.array(seq_count)


def get_ngram_sequences(ngram_array):
    cols_seq = []
    rows_seq = []
    for y in range(ngram_array.shape[0]):
        rows_seq.append(get_row_sequences(ngram_array[y,:]))
    for x in range(ngram_array.shape[1]):
        cols_seq.append(get_row_sequences(ngram_array[:,x]))
    return cols_seq, rows_seq


def get_squares_width_and_margins(ngram_array, margin):
    horizontal_margin, vertical_margin, squares_width = None, None, None
    if (ngram_array.shape[1]/ngram_array.shape[0]) >= (win_width/win_height):
        horizontal_margin = margin
        squares_width = (win_width - 2*margin) / ngram_array.shape[1]
        vertical_margin = (win_height - (squares_width * ngram_array.shape[0])) / 2
    else:
        vertical_margin = margin
        squares_width = (win_height - 2*margin) / ngram_array.shape[0]
        horizontal_margin = (win_width - (squares_width * ngram_array.shape[1])) / 2

    return horizontal_margin, vertical_margin, squares_width


def draw_ngram_puzzle(ngram_array, margin, solved, cols_seq=None, rows_seq=None):
    horizontal_margin, vertical_margin, squares_width = get_squares_width_and_margins(ngram_array, margin)

    # fill frame with white
    win.fill((255,255,255))

    # draw the numbers
    font = pygame.font.Font(None, int(squares_width))
    for x in range(ngram_array.shape[1]):
        sequences = cols_seq[x] if cols_seq is not None else get_row_sequences(ngram_array[:, x])
        for idx,count in enumerate(sequences[::-1]):
            draw_text(str(count), (horizontal_margin+x*squares_width+squares_width/4, vertical_margin-idx*squares_width-squares_width), font, (0,0,0), disp_update=False)
    for y in range(ngram_array.shape[0]):
        sequences = rows_seq[y] if rows_seq is not None else get_row_sequences(ngram_array[y, :])
        for idx,count in enumerate(sequences[::-1]):
            draw_text(str(count), (horizontal_margin-idx*squares_width-squares_width, vertical_margin+y*squares_width+squares_width/4), font, (0,0,0), disp_update=False)

    # draw squares
    if solved:
        for x in range(ngram_array.shape[1]):
            for y in range(ngram_array.shape[0]):
                if ngram_array[y,x]:
                    pygame.draw.rect(win, (50,50,50), (horizontal_margin+x*squares_width,vertical_margin+y*squares_width,squares_width+1,squares_width+1))

    # draw lines
    for x in range(ngram_array.shape[1]+1):
        width = 2 if x % 5 == 0 else 1
        pygame.draw.line(win, (0,0,0), (horizontal_margin+x*squares_width,0), (horizontal_margin+x*squares_width,vertical_margin+squares_width*ngram_array.shape[0]), width=width)
    for y in range(ngram_array.shape[0]+1):
        width = 2 if y % 5 == 0 else 1
        pygame.draw.line(win, (0, 0, 0), (0, vertical_margin + y * squares_width), (horizontal_margin + squares_width * ngram_array.shape[1], vertical_margin + y * squares_width), width=width)

    pygame.display.update()


def create_puzzle_from_img(img_file):
    global img_arr
    if pixelization_coef > 0:
        img_arr = convert_img_file_to_bool_arr(img_file, pixelization_coef=pixelization_coef)
    img_arr = img_arr != 1 if invert_colors else img_arr
    draw_ngram_puzzle(img_arr, margin, is_solved)


def draw_colored_rect_on_creating_removing_place():
    disp_and_ngram_update = False
    horizontal_margin, vertical_margin, squares_width = get_squares_width_and_margins(img_arr, margin)
    # draw red rectangle on the row/column that is about to be removed
    # left column
    if horizontal_margin < pygame.mouse.get_pos()[0] < horizontal_margin + squares_width \
            and pygame.mouse.get_pos()[1] < vertical_margin and is_in_left_click:
        pygame.draw.rect(win, (100, 0, 0), (horizontal_margin, 0, squares_width, vertical_margin))
        disp_and_ngram_update = True
    # right column
    if win_width - horizontal_margin - squares_width < pygame.mouse.get_pos()[0] < win_width - horizontal_margin \
            and pygame.mouse.get_pos()[1] < vertical_margin and is_in_left_click:
        pygame.draw.rect(win, (100, 0, 0),
                         (win_width - horizontal_margin - squares_width, 0, squares_width, vertical_margin))
        disp_and_ngram_update = True
    # top row
    if vertical_margin < pygame.mouse.get_pos()[1] < vertical_margin + squares_width \
            and pygame.mouse.get_pos()[0] < horizontal_margin and is_in_left_click:
        pygame.draw.rect(win, (100, 0, 0), (0, vertical_margin, horizontal_margin, squares_width))
        disp_and_ngram_update = True
    # bottom row
    if win_height - vertical_margin - squares_width < pygame.mouse.get_pos()[1] < win_height - vertical_margin \
            and pygame.mouse.get_pos()[0] < horizontal_margin and is_in_left_click:
        pygame.draw.rect(win, (100, 0, 0),
                         (0, win_height - vertical_margin - squares_width, horizontal_margin, squares_width))
        disp_and_ngram_update = True

    # draw green rectangle on the position where a row/column would be created
    # left column
    if horizontal_margin - squares_width < pygame.mouse.get_pos()[0] < horizontal_margin \
            and pygame.mouse.get_pos()[1] < vertical_margin and is_in_left_click:
        pygame.draw.rect(win, (0, 100, 0), (horizontal_margin - squares_width, 0, squares_width, vertical_margin))
        disp_and_ngram_update = True
    # right column
    if win_width - horizontal_margin < pygame.mouse.get_pos()[0] < win_width - horizontal_margin + squares_width \
            and pygame.mouse.get_pos()[1] < vertical_margin and is_in_left_click:
        pygame.draw.rect(win, (0, 100, 0), (win_width - horizontal_margin, 0, squares_width, vertical_margin))
        disp_and_ngram_update = True
    # top row
    if vertical_margin - squares_width < pygame.mouse.get_pos()[1] < vertical_margin \
            and pygame.mouse.get_pos()[0] < horizontal_margin and is_in_left_click:
        pygame.draw.rect(win, (0, 100, 0), (0, vertical_margin - squares_width, horizontal_margin, squares_width))
        disp_and_ngram_update = True
    # bottom row
    if win_height - vertical_margin < pygame.mouse.get_pos()[1] < win_height - vertical_margin + squares_width \
            and pygame.mouse.get_pos()[0] < horizontal_margin and is_in_left_click:
        pygame.draw.rect(win, (0, 100, 0), (0, win_height - vertical_margin, horizontal_margin, squares_width))
        disp_and_ngram_update = True

    # check if need to update display and ngram
    if disp_and_ngram_update:
        pygame.display.update()
        draw_ngram_puzzle(img_arr, margin, is_solved)


def add_or_remove_row_col_in_mouse_pos():
    global img_arr
    horizontal_margin, vertical_margin, squares_width = get_squares_width_and_margins(img_arr, margin)
    # remove rows/columns that marked with red
    # left column
    if horizontal_margin < pygame.mouse.get_pos()[0] < horizontal_margin + squares_width and pygame.mouse.get_pos()[1] < vertical_margin:
        img_arr = np.delete(img_arr, 0, 1)
        draw_ngram_puzzle(img_arr, margin, is_solved)
    # right column
    if win_width - horizontal_margin - squares_width < pygame.mouse.get_pos()[0] < win_width - horizontal_margin and pygame.mouse.get_pos()[1] < vertical_margin:
        img_arr = np.delete(img_arr, -1, 1)
        draw_ngram_puzzle(img_arr, margin, is_solved)
    # top row
    if vertical_margin < pygame.mouse.get_pos()[1] < vertical_margin + squares_width and pygame.mouse.get_pos()[0] < horizontal_margin:
        img_arr = np.delete(img_arr, 0, 0)
        draw_ngram_puzzle(img_arr, margin, is_solved)
    # bottom row
    if win_height - vertical_margin - squares_width < pygame.mouse.get_pos()[1] < win_height - vertical_margin and pygame.mouse.get_pos()[0] < horizontal_margin:
        img_arr = np.delete(img_arr, -1, 0)
        draw_ngram_puzzle(img_arr, margin, is_solved)

    # draw green rectangle on the position where a row/column would be created
    # left column
    if horizontal_margin - squares_width < pygame.mouse.get_pos()[0] < horizontal_margin and pygame.mouse.get_pos()[1] < vertical_margin:
        img_arr = np.column_stack((np.zeros(img_arr.shape[0]), img_arr))
        draw_ngram_puzzle(img_arr, margin, is_solved)
    # right column
    if win_width - horizontal_margin < pygame.mouse.get_pos()[0] < win_width - horizontal_margin + squares_width and pygame.mouse.get_pos()[1] < vertical_margin:
        img_arr = np.column_stack((img_arr, np.zeros(img_arr.shape[0])))
        draw_ngram_puzzle(img_arr, margin, is_solved)
    # top row
    if vertical_margin - squares_width < pygame.mouse.get_pos()[1] < vertical_margin and pygame.mouse.get_pos()[0] < horizontal_margin and is_in_left_click:
        img_arr = np.row_stack((np.zeros(img_arr.shape[1]), img_arr))
        draw_ngram_puzzle(img_arr, margin, is_solved)
    # bottom row
    if win_height - vertical_margin < pygame.mouse.get_pos()[1] < win_height - vertical_margin + squares_width and pygame.mouse.get_pos()[0] < horizontal_margin:
        img_arr = np.row_stack((img_arr, np.zeros(img_arr.shape[1])))
        draw_ngram_puzzle(img_arr, margin, is_solved)


def add_or_remove_square_in_mouse_pos():
    horizontal_margin, vertical_margin, squares_width = get_squares_width_and_margins(img_arr, margin)
    if 0 <= int((pygame.mouse.get_pos()[0] - horizontal_margin) // squares_width) < img_arr.shape[1] \
            and 0 <= int((pygame.mouse.get_pos()[1] - vertical_margin) // squares_width) < img_arr.shape[0]:
        if is_in_left_click:
            img_arr[int((pygame.mouse.get_pos()[1] - vertical_margin) // squares_width), int(
                (pygame.mouse.get_pos()[0] - horizontal_margin) // squares_width)] = 1
        if is_in_right_click:
            img_arr[int((pygame.mouse.get_pos()[1] - vertical_margin) // squares_width), int(
                (pygame.mouse.get_pos()[0] - horizontal_margin) // squares_width)] = 0
        draw_ngram_puzzle(img_arr, margin, is_solved)


def save_screen():
    try:
        squares_width = get_squares_width_and_margins(img_arr, margin)[2]
        img_file_name = filedialog.asksaveasfilename()
        font = pygame.font.Font('gan/ganclm_bold-webfont.ttf', int(squares_width) * 2)
        text = img_file_name
        if img_file_name.find('/') != -1:
            text = text[(len(text) - text[::-1].find('/')):]
        if img_file_name.find('.') != -1:
            text = text[:(len(text) - text[::-1].find('.'))-1]
        text = text[::-1]
        draw_text(text, (100, 100), font, (0, 0, 0))
        pygame.image.save(win, f'{img_file_name}.jpg' if '.' not in img_file_name else img_file_name)
    except Exception as e:
        logging.error('Failed saving file')
        logging.error(e)
        font = pygame.font.Font(None, 150)
        draw_text('Failed saving file', (win_width / 2 - 500, win_height // 2 - 75), font, (255, 0, 0))
        pygame.display.update()


def stop_until_key_pressed(key, key2: 'optional' = None):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == key or event.key == key2:
                    return


def get_options(row_len, sequences):
    options = []
    for order in permutations(np.append(np.ones(sequences.shape[0]), np.zeros(row_len-sequences.shape[0]))):
        option = np.zeros(row_len).astype(int)
        replacement_indices = np.where(order==1)[0] + np.append([0], np.cumsum(sequences)[:-1])
        for length,position in zip(sequences,replacement_indices):
            option[position:position+length] = 1
        options.append(option)
    return options


def get_overlaps(row_len, sequences):
    overlaps = np.ones(row_len).astype(int)
    for option in get_options(row_len, sequences):
        overlaps = np.bitwise_and(option, overlaps)
    return overlaps


def solve_row(row, sequences):
    overlaps = get_overlaps(row.shape[0], sequences)
    row[overlaps==1] = 1


def solve(cols_seq:list, rows_seq:list):
    ngram_arr = np.zeros((len(rows_seq), len(cols_seq)))
    is_changed = True
    while is_changed:
        is_changed = False
        for y in range(ngram_arr.shape[0]):
            stop_until_key_pressed(pygame.K_SPACE)
            row_before_solve = ngram_arr[y,:]
            solve_row(row_before_solve, rows_seq[y])
            if not np.array_equal(row_before_solve, ngram_arr[y,:]):
                is_changed = True
            draw_ngram_puzzle(ngram_arr, margin, True, cols_seq, rows_seq)
        for x in range(ngram_arr.shape[1]):
            stop_until_key_pressed(pygame.K_SPACE)
            col_before_solve = ngram_arr[:,x]
            solve_row(col_before_solve, cols_seq[x])
            if not np.array_equal(col_before_solve, ngram_arr[:,x]):
                is_changed = True
            draw_ngram_puzzle(ngram_arr, margin, True, cols_seq, rows_seq)


'''def calculate_options_num(row, sequences):
    return factorial(row.shape[0] - np.sum(sequences) + 1) / (factorial(sequences.shape[0]) * factorial(row.shape[0] - sequences.shape[0] - np.sum(sequences) + 1))


def solve(ngram_solved_part, cols_seq, rows_seq):
    global Mode
    Mode = solve_mode
    draw_ngram_puzzle(ngram_solved_part, margin, True, Mode, cols_seq=cols_seq, rows_seq=rows_seq)
    lowest_options_num_row_idx = 0
    for y in range(ngram_solved_part.shape[0]):
        if calculate_options_num(ngram_solved_part[lowest_options_num_row_idx,:]) < lowest_options_num_row_idx:
            lowest_options_num_row_idx = y'''



create_mode = 'create'
solve_mode = 'solve'
Mode = solve_mode


is_solved = True
is_in_left_click = False
is_in_right_click = False
pixelization_coef = 15
invert_colors = True
threshold_addition = 0
margin = 370
img_file_name = 'images/test8.jpg'
img_arr = convert_img_file_to_bool_arr(img_file_name, pixelization_coef=pixelization_coef)
create_puzzle_from_img(img_file_name)

if Mode == solve_mode:
    cols_seq, rows_seq = get_ngram_sequences(img_arr)
    solve(cols_seq, rows_seq)

while run:
    draw_colored_rect_on_creating_removing_place()
    if is_in_left_click or is_in_right_click:
        add_or_remove_square_in_mouse_pos()

    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_in_left_click = True
            if event.button == 3:
                is_in_right_click = True

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                add_or_remove_row_col_in_mouse_pos()
                is_in_left_click = False

            if event.button == 3:
                is_in_right_click = False
            if event.button == 4:
                # scroll forward
                if margin > 20:
                    margin -= 20
                draw_ngram_puzzle(img_arr, margin, is_solved)
            if event.button == 5:
                # scroll back
                if margin < (win_height//2 - 100):
                    margin += 20
                draw_ngram_puzzle(img_arr, margin, is_solved)

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s and pygame.key.get_mods() & pygame.KMOD_CTRL:
                save_screen()
            if event.key == pygame.K_SPACE:
                is_solved = not is_solved
                draw_ngram_puzzle(img_arr, margin, is_solved)

            if event.key == pygame.K_UP:
                threshold_addition += 0.01
                create_puzzle_from_img(img_file_name)
            if event.key == pygame.K_DOWN:
                threshold_addition -= 0.01
                create_puzzle_from_img(img_file_name)

            if event.key == pygame.K_RIGHT:
                pixelization_coef += 1
                create_puzzle_from_img(img_file_name)
            if event.key == pygame.K_LEFT:
                if pixelization_coef > 1:
                    pixelization_coef -= 1
                create_puzzle_from_img(img_file_name)

            if event.key == pygame.K_o:
                threshold_addition = 0
                try:
                    img_file_name = filedialog.askopenfilename()
                    create_puzzle_from_img(img_file_name)
                except Exception as e:
                    logging.error('Failed uploading file')
                    logging.error(e)
                    font = pygame.font.Font(None, 150)
                    draw_text('Failed uploading file', (win_width/2-500,win_height//2-75), font, (255,0,0))
                    pygame.display.update()

            if event.key == pygame.K_i:
                img_arr = np.invert(img_arr)
                draw_ngram_puzzle(img_arr, margin, is_solved)
            if event.key == pygame.K_ESCAPE:
                run = False
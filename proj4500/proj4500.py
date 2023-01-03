# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:27:49 2022

@author: kl001
"""
from __future__ import print_function
import numpy as np
import os
import time
from contextlib import contextmanager
from math import floor
import usb.core
import usb.util
from usb.core import USBError
from time import perf_counter_ns
import sys
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test\proj4500')
sys.path.append(r'C:\Users\kl001\Documents\pyfringe_test')
import FringeAcquisition as fa
import cv2
import PySpin



def conv_len(a, l):
    """
    Function that converts a number into a bit string of given length.

    :param int a: Number to convert.
    :param int l: Length of bit string.

    :return: Padded bit string.
    """
    b = bin(a)[2:]
    padding = l - len(b)
    b = '0' * padding + b
    return b


def bits_to_bytes(a, reverse=True): # default is reverse
    """
    Function that converts bit string into a given number of bytes.

    :param str a: Bytes to convert.
    :param bool reverse: Whether or not to reverse the byte list.

    :return: List of bytes.
    """
    bytelist = []

    # check if needs padding
    if len(a) % 8 != 0:
        padding = 8 - len(a) % 8
        a = '0' * padding + a

    # convert to bytes
    for i in range(len(a) // 8):
        bytelist.append(int(a[8 * i:8 * (i + 1)], 2))

    if reverse:
        bytelist.reverse()
    return bytelist


def fps_to_period(fps):
    """
    Calculates desired period (us) from given fps.

    :param int fps: Frames per second.

    :return: Period (us).
    """
    period = int(floor(1.0 / fps * 10**6))
    return period

@contextmanager
def connect_usb():
    """
    Context manager for connecting to and releasing usb device.

    :yields: USB device.
    """
    device = usb.core.find(idVendor=0x0451, idProduct=0x6401) #finding the projector usb port
    device.set_configuration()

    lcr = dlpc350(device)

    yield lcr

    device.reset()
    del lcr
    del device

class dlpc350(object):
    """
    Class representing dmd controller. Can connect to different DLPCs by changing product ID. Check IDs in
    device manager.
    """
    def __init__(self, device):
        """
        Connects the device.

        :param device: lcr4500 USB device.
        """
        self.dlpc = device
        self.mode = 'pattern' 
        self.source = 'flash'
        self.exposure_period = None
        self.frame_period = None
        
    def command(self,
                rw_mode,
                sequence_byte,
                com1,
                com2,
                data=None):
        """
        Sends a command to the dlpc.

        :param str rw_mode: Whether reading or writing.
        :param int sequence_byte:
        :param int com1: Command 1
        :param int com2: Command 3
        :param list data: Data to pass with command.
        """

        buffer = []

        if rw_mode == 'r':
            flagstring = 0xc0  # 0b11000000
        else:
            flagstring = 0x40  # 0b01000000

        data_len = conv_len(len(data) + 2, 16) # size of data + subcommands to 16 bits binary
        data_len = bits_to_bytes(data_len)

        buffer.append(flagstring)
        buffer.append(sequence_byte)
        buffer.extend(data_len)
        buffer.append(com2)
        buffer.append(com1)

        # # if data fits into single buffer, write all and fill. Single command = 64 bytes
        if len(buffer) + len(data) < 65:
            for i in range(len(data)):
                buffer.append(data[i])

            # append empty data to fill buffer
            for i in range(64 - len(buffer)):
                buffer.append(0x00)

            self.dlpc.write(1, buffer)

        # else, keep filling buffer and pushing until data all sent
        else:
            for i in range(64 - len(buffer)):
                buffer.append(data[i])

            self.dlpc.write(1, buffer)
            buffer = []

            j = 0
            while j < len(data) - 58:
                buffer.append(data[j + 58])
                j += 1

                if j % 64 == 0:
                    self.dlpc.write(1, buffer)
                    buffer = []

            if j % 64 != 0:
                while j % 64 != 0:
                    buffer.append(0x00)
                    j += 1

                self.dlpc.write(1, buffer)

        
        try:
            self.ans = self.dlpc.read(0x81, 64)            
            length_lsb = self.ans[2]
            length_msb = self.ans[3]
            message_length = length_msb*256 + length_lsb
            num_packet = message_length//64 + 1
            if num_packet > 1:
                for i in range(num_packet-1):
                    self.ans.extend(self.dlpc.read(0x81, 64))                
        except USBError as e:
            print('USB Error:', e)

        time.sleep(0.02)

    def read_reply(self):
        """
        Reads in reply.
        """
        for i in self.ans:
            print(hex(i))

    def get_main_status(self, pretty_print=False):
        """The Main Status command shows the status of DMD park and DLPC350 sequencer, frame buffer, and gamma
         correction.

         (USB: CMD2: 0x02, CMD3: 0x0C)
         """
        self.command('r', 0x00, 0x1a, 0x0c, [])  # rw_mode, sequence,com1,com2=0x0c for main status, data
        if pretty_print:
            # ans = str(bin(self.ans[4]))[2:]
            ans = format(self.ans[4], '08b')
            print(f'\nDMD micromirrors are {"parked" if int(ans[-1]) else "not parked"}')
            print(f'Sequencer is {"running normally" if int(ans[-2]) else "stopped"}')
            print(f'Frame buffer is {"frozen" if int(ans[-3]) else "not frozen"}')
            print(f'Gamma correction is {"enabled" if int(ans[-4]) else "disabled"}')
            
    def retrieve_flashimages(self):
        """
        This command retrieves the information about the number of Images in the flash.
        
        """
        
        self.command('r', 0x00, 0x1a, 0x42,[])
        ans = self.ans[4]
        print(f'No. of images in the flash:{ans}')
        
    def read_mode(self):  #default mode is 'pattern'
        """
        Read the current  input mode for the projector.

        (USB: CMD2: 0x1A, CMD3: 0x1B)
        """
        self.command('r', 0x00, 0x1a, 0x1b, [])
        ans = bin(self.ans[4]) 
        print(f'Current mode:{"pattern" if int(ans[-1]) else "video"}')
    
    def set_display_mode(self, mode):  #default mode is 'pattern'
        """
        Selects the input mode for the projector.

        (USB: CMD2: 0x1A, CMD3: 0x1B)

        :param int mode:
            :0: "video" mode
            :1: "pattern" mode
        """
        self. mode = mode
        modes = ['video', 'pattern'] #video = 0, pattern =1
        if mode in modes:
            mode = modes.index(mode)

        self.command('w', 0x00, 0x1a, 0x1b, [mode])
        
    def read_pattern_input_source(self):
        """
        Read current input source.
        """
        self.command('r', 0x00, 0x1a, 0x22, [])
        ans = bin(self.ans[4])
        print(f'Current input source:{"video" if ans[-2:] == 11 else "flash"}')
        
    def set_pattern_input_source(self, source='video'):  # pattern source default = 'video'
        """
        Selects the input type for pattern sequence.

        (USB: CMD2: 0x1A, CMD3: 0x22)

        :param int mode:
            :0: "video"
            :3: "flash"
        """
        self.source = source
        sources = ['video', '', '', 'flash']  # video = 0, reserved, reserved, flash=11 (bin 3)
        if source in sources:
            source = sources.index(source)

        self.command('w', 0x00, 0x1a, 0x22, [source])
        
    
    def read_pattern_config(self):
        """
        Read current pattern loopup table.
        """
        self.command('r', 0x00, 0x1a, 0x31, [])
        ans = self.ans[4:8]
        print(f'Current config:\n \t No. of LUT entries : {int(ans[-4]) + 1} \n \t Repeat sequence: {"yes" if int(ans[-3]) else "no"}')
        print(f'\t No. Number of patterns to display:{ int(ans[-2])+1} \n \t No. of image index:{ int(ans[-1])+1} ')
       
        
    def set_pattern_config(self,
                           num_lut_entries=15,
                           do_repeat=False, # Default repeat pattern 
                           num_pats_for_trig_out2= 15,
                           num_images=5):
        """
        This API controls the execution of patterns stored in the lookup table. Before using this API, stop the current
        pattern sequence using ``DLPC350_PatternDisplay()`` API. After calling this API, send the Validation command
        using the API DLPC350_ValidatePatLutData() before starting the pattern sequence.

        (USB: CMD2: 0x1A, CMD3: 0x31)

        :param int num_lut_entries: Number of LUT entries.
        :param bool do_repeat:

            :True: Execute the pattern sequence once.
            :False: Repeat the pattern sequence.

        :param int num_pats_for_trig_out2: Number of patterns to display(range 1 through 256). If in repeat mode, then
           this value dictates how often TRIG_OUT_2 is generated.

        :param int num_images: Number of Image Index LUT Entries(range 1 through 64). This Field is irrelevant for Pattern
            Display Data Input Source set to a value other than internal.
        """
        num_lut_entries = '0' + conv_len(num_lut_entries - 1, 7)
        do_repeat = '0000000' + str(int(do_repeat))
        num_pats_for_trig_out2 = conv_len(num_pats_for_trig_out2 - 1, 8)
        num_images = '00' + conv_len(num_images - 1, 6)

        payload = num_images + num_pats_for_trig_out2 + do_repeat + num_lut_entries
        payload = bits_to_bytes(payload)

        self.command('w', 0x00, 0x1a, 0x31, payload)
    
    def read_pattern_trigger_mode(self):
        """
        Read current trigger mode.
        """
        self.command('r', 0x00, 0x1a, 0x23, [])
        ans = self.ans[4]
        print('Current trigger mode:{}'.format(ans))
        
    def set_pattern_trigger_mode(self, mode='vsync'):
        """
        Selects the trigger type for pattern sequence.

        (USB: CMD2: 0x1A, CMD3: 0x23)

        :param int mode:
            :0: "vsync"
        """
        modes = ['vsync','trig_mode1', 'trig_mode2', 'trig_mode3', 'trig_mode4']
        if mode in modes:
            mode = modes.index(mode)

        self.command('w', 0x00, 0x1a, 0x23, [mode])
    
    def read_trig_out1_control(self):
        """
        Read current trig_out1 setting.
        """
        self.command('r', 0x00, 0x1a, 0x1d, [])
        ans = self.ans[4:7]
        print('Current trig_out1 setting:')
        print(f'\t Polarity: {"active low signal" if int(ans[0]) else "active high signal"}')
        print(f'\t Rising edge delay:{ans[1]}')
        print(f'\t Falling edge delay:{ans[-1]}')
       
        
    def trig_out1_control(self,plarity_invert = True, trigedge_rise_delay = 187, trigedge_fall_delay = 187):
         """
         The Trigger Out1 Control command sets the polarity, rising edge delay, 
         and falling edge delay of the TRIG_OUT_1 signal of the DLPC350. Before executing this command, stop the current pattern sequence. 
         After executing this command, send the Validation command (I2C: 0x7D or USB: 0x1A1A) once before starting the pattern sequence.
         param bool plarity_invert: True for active low signal
         param int trigedge_rise_delay: rising edge delay control ranging from –20.05 μs to 2.787 μs. Each bit adds 107.2 ns.
         param int trigedge_fall_delay: falling edge delay control with range -20.05 μs to +2.787 μs. Each bit adds 107.2 ns
         """
         if plarity_invert:
             polarity = '00000010'
         else:
             polarity = '00000000'
         trigedge_rise_delay = conv_len(trigedge_rise_delay, 8)
         trigedge_fall_delay = conv_len(trigedge_fall_delay, 8)
         
         payload =   trigedge_fall_delay + trigedge_rise_delay + polarity
         payload = bits_to_bytes(payload)
         
         self.command('w', 0x00, 0x1a, 0x1d, payload)
         
    def read_exposure_frame_period(self):
        """
        Read current exposure time and frame period
        """
        self.command('r', 0x00, 0x1a, 0x29, [])
        ans = self.ans
        exposure_time = ans[4] + ans[5]*256 + ans[6]*256**2 + ans[7]*256**3
        frame_period = ans[8] + ans[9]*256 + ans[10]*256**2 + ans[11]*256**3
        
        print('Pattern exposure time:%f'%exposure_time)
        print('Frame period:%f'%frame_period)
    
    def set_exposure_frame_period(self, exposure_period, frame_period):
        """
        The Pattern Display Exposure and Frame Period dictates the time a pattern is exposed and the frame period.
        Either the exposure time must be equivalent to the frame period, or the exposure time must be less than the
        frame period by 230 microseconds. Before executing this command, stop the current pattern sequence. After
        executing this command, call ``DLPC350_ValidatePatLutData()`` API before starting the pattern sequence.

        (USB: CMD2: 0x1A, CMD3: 0x29)

        :param int exposure_period: Exposure time in microseconds (4 bytes).
        :param int frame_period: Frame period in microseconds (4 bytes).
        """
        self.exposure_period = exposure_period
        self.frame_period = frame_period
        exposure_period = conv_len(exposure_period, 32) # decimal to bit string of size 32
        frame_period = conv_len(frame_period, 32) # decimal to bit string of size 32

        payload = frame_period + exposure_period
        payload = bits_to_bytes(payload) # it will be reverse

        self.command('w', 0x00, 0x1a, 0x29, payload)
  
        
    def start_pattern_lut_validate(self):
         """
         This API checks the programmed pattern display modes and indicates any invalid settings. This command needs to
         be executed after all pattern display configurations have been completed.

         (USB: CMD2: 0x1A, CMD3: 0x1A)
         """
         self.command('w', 0x00, 0x1a, 0x1a, bits_to_bytes(conv_len(0x00, 8))) # Pattern Display Mode: Validate Data: CMD2: 0x1A, CMD3: 0x1A
    
    def read_lut_validate(self):
        """
        Read validation
        """
        self.command('r', 0x00, 0x1a, 0x1a,[])
        ans = conv_len(self.ans[4],8) # STR        
        print('Validation result\n')
        print(f'Exposure and frame period setting: {"invalid" if int(ans[-1]) else "valid"}\n')
        print(f'LUT: {"invalid" if int(ans[-2]) else "valid"}\n')
        print(f'Trigger Out1: {"invalid" if int(ans[-3]) else "valid"}\n')
        print(f'Post sector settings: {"invalid" if int(ans[-4]) else "valid"}\n')
        print(f'DLPC350 is {"busy" if int(ans[-8]) else "valid"}\n')
        
        return ans
       
        
    def open_mailbox(self, mbox_num):
        """
        This API opens the specified Mailbox within the DLPC350 controller. This API must be called before sending data
        to the mailbox/LUT using DLPC350_SendPatLut() or DLPC350_SendImageLut() APIs.

        (USB: CMD2: 0x1A, CMD3: 0x33)

        :param mbox_num:
            :0: Disable (close) the mailboxes.
            :1: Open the mailbox for image index configuration.
            :2: Open the mailbox for pattern definition.
            :3: Open the mailbox for the Variable Exposure.
        """
        mbox_num = bits_to_bytes(conv_len(mbox_num, 8))
        self.command('w', 0x00, 0x1a, 0x33, mbox_num)
        
    def mailbox_set_address(self, address=0):
        """
        This API defines the offset location within the DLPC350 mailboxes to write data into or to read data from.

        (USB: CMD2: 0x1A, CMD3: 0x32)

        :param int address: Defines the offset within the selected (opened) LUT to write/read data to/from (0-127).
        """
        address = bits_to_bytes(conv_len(address, 8))
        self.command('w', 0x00, 0x1a, 0x32, address)
        
    def read_mailbox_address(self):
        self.command('r', 0x00, 0x1a, 0x32, [])
        return self.ans.tolist()
        
    def pattern_flash_index(self, index):
        """
        The following parameters: display mode, trigger mode, exposure, and frame rate must be set up before sending any mailbox data.
        If the mailbox is opened to define the flash image indexes, list the index numbers in the mailbox. 
        For example, if image indexes 0 through 3 are desired, write 0x0 0x1 0x2 0x3 to the mailbox. 
        
        :param index: image index to be written
        """
        index = bits_to_bytes(conv_len(index, 8))        
        self.command('w', 0x00, 0x1a, 0x34, index)
    
    def send_pattern_lut(self,
                         trig_type,
                         pat_num,
                         bit_depth,
                         led_select,do_buf_swap,
                         do_invert_pat=False,
                         do_insert_black=True,
                         do_trig_out_prev=False):
        """
        Mailbox content to setup pattern definition. See table 2-65 in programmer's guide for detailed description of
        pattern LUT entries.

        (USB: CMD2: 0x1A, CMD3: 0x34)

        :param int trig_type: Select the trigger type for the pattern

            :0: Internal trigger.
            :1: External positive.
            :2: External negative.
            :3: No Input Trigger (Continue from previous; Pattern still has full exposure time).
            :0x3FF: Full Red Foreground color intensity

        :param int pat_num: Pattern number (0 based index). For pattern number ``0x3F``, there is no pattern display. The
            maximum number supported is 24 for 1 bit-depth patterns. Setting the pattern number to be 25, with a
            bit-depth of 1 will insert a white-fill pattern. Inverting this pattern will insert a black-fill pattern.
            These patterns will have the same exposure time as defined in the Pattern Display Exposure and Frame Period
            command. Table 2-70 in the programmer's guide illustrates which bit planes are illuminated by each pattern
            number.

        :param bit_depth: Select desired bit-depth

            :0: Reserved
            :1: 1-bit
            :2: 2-bit
            :3: 3-bit
            :4: 4-bit
            :5: 5-bit
            :6: 6-bit
            :7: 7-bit
            :8: 8-bit

        :param int led_select: Choose the LEDs that are on (bit flags b0 = Red, b1 = Green, b2 = Blue)

            :0: 0b000 No LED (Pass through)
            :1: 0b001 Red
            :2: 0b010 Green
            :3: 0b011 Yellow (Green + Red)
            :4: 0b100 Blue
            :5: 0b101 Magenta (Blue + Red)
            :6: 0b110 Cyan (Blue + Green)

        :param bool do_invert_pat:
            :True: Invert pattern.
            :False: Do not invert pattern.

        :param bool do_insert_black:
            :True: Insert black-fill pattern after current pattern. This setting requires 230 us of time before the
               start of the next pattern.
            :False: Do not insert any post pattern.

        :param bool do_buf_swap:
            :True: perform a buffer swap.
            :False: Do not perform a buffer swap.

        :param do_trig_out_prev:
            :True: Trigger Out 1 will continue to be high. There will be no falling edge between the end of the
               previous pattern and the start of the current pattern. Exposure time is shared between all patterns
               defined under a common trigger out). This setting cannot be combined with the black-fill pattern.
            :False: Trigger Out 1 has a rising edge at the start of a pattern, and a falling edge at the end of the
               pattern.

        """
        # byte 0
        trig_type = conv_len(trig_type, 2)
        pat_num = conv_len(pat_num, 6)

        byte_0 = pat_num + trig_type

        # byte 1
        bit_depth = conv_len(bit_depth, 4)
        led_select = conv_len(led_select, 4)

        byte_1 = led_select + bit_depth
        

        # byte 2
        do_invert_pat = str(int(do_invert_pat))
        do_insert_black = str(int(do_insert_black))
        do_buf_swap = str(int(do_buf_swap))
        do_trig_out_prev = str(int(do_trig_out_prev))

        byte_2 = '0000' + do_trig_out_prev + do_buf_swap + do_insert_black + do_invert_pat

        payload = byte_2 + byte_1 + byte_0
        payload = bits_to_bytes(payload)

        self.command('w', 0x00, 0x1a, 0x34, payload)

    def pattern_display(self, action='start'):
         """
         This API starts or stops the programmed patterns sequence.

         (USB: CMD2: 0x1A, CMD3: 0x24)

         :param int action: Pattern Display Start/Stop Pattern Sequence

             :0: Stop Pattern Display Sequence. The next "Start" command will restart the pattern sequence from the
                beginning.
             :1: Pause Pattern Display Sequence. The next "Start" command will start the pattern sequence by
                re-displaying the current pattern in the sequence.
             :2: Start Pattern Display Sequence.
         """
         actions = ['stop', 'pause', 'start']
         if action in actions:
             action = actions.index(action)

         self.command('w', 0x00, 0x1a, 0x24, [action])
         
    def read_mailbox_info(self):
        self.command('r', 0x00, 0x1a, 0x34, [])
        ans =self.ans[4:]
        print('Image index:{}'.format(ans))
        return ans


def get_image_LUT_swap_location(image_index_list):
    swap_location_list = [0] + [i for i in range(1,len(image_index_list)) if image_index_list[i]!=image_index_list[i-1]]
    image_LUT_entries = [image_index_list[i] for i in swap_location_list]

    if len(image_LUT_entries) == 2:    
        temp = image_LUT_entries[0:2].copy()
        image_LUT_entries[0] = temp[1]
        image_LUT_entries[1] = temp[0]
        
    return image_LUT_entries, swap_location_list

def new_LUT_validation(image_index_list, swap_location_list,  image_LUT_entries_read, lut_read):    
    
    swap_location_list_read = []
    for i,k in enumerate(lut_read[2::3]):
        if k == 6:
            swap_location_list_read.append(i)   
    
    num_of_patterns = len(image_index_list)
    if len(image_LUT_entries_read) == 2:
        image_LUT_entries_temp = image_LUT_entries_read.copy()
        temp = image_LUT_entries_temp[0:2].copy()
        image_LUT_entries_temp[0] = temp[1]
        image_LUT_entries_temp[1] = temp[0]
    else:
        image_LUT_entries_temp = image_LUT_entries_read

    num_repeats = [swap_location_list_read[i] - swap_location_list_read[i-1] for i in range(1,len(swap_location_list_read))] + [num_of_patterns - swap_location_list_read[-1]]
    image_index_list_recovered = []
    for i,num in enumerate(num_repeats):
        image_index_list_recovered.extend([image_LUT_entries_temp[i]]*num)
        
    if image_index_list == image_index_list_recovered:
        print("Recovery successfull")
    else:
        print("Recovery test failed")
    
    return image_index_list_recovered, swap_location_list_read

def current_setting():
    with connect_usb() as lcr:

        # before proceeding to change params, need to stop pattern sequence mode
        lcr.pattern_display('stop')
        
        # Get current status 
        lcr.get_main_status(pretty_print=True)
        # Current mode
        lcr.read_mode()
        #number of flash images
        lcr.retrieve_flashimages()
        #pattern source
        lcr.read_pattern_input_source()
        #pattern config
        lcr.read_pattern_config()
        #pattern trigger mode
        lcr.read_pattern_trigger_mode()
        #trigger 1 parameters (polarity,delays)
        lcr.read_trig_out1_control()
        #exposure and frame period
        lcr.read_exposure_frame_period()
        # get current image index
        lcr.open_mailbox(1)
        lcr.mailbox_set_address(address = 0)
        image_LUT_entries_read = lcr.read_mailbox_info().tolist()        
        lcr.open_mailbox(0)
        #get current LUT
        lcr.open_mailbox(2)
        lcr.mailbox_set_address(address = 0)
        lut_read = lcr.read_mailbox_info().tolist()
        # current_address = lcr.read_mailbox_address()
        
        lcr.open_mailbox(0)
        
        
        #============================================================
        # start validation
        lcr.start_pattern_lut_validate()
        #Check validation status
        ret = 1
        start = perf_counter_ns()
        while ret:
            ans = lcr.read_lut_validate()
            ret = int(ans[0])    
            end = perf_counter_ns()
            t = (end - start)/1e9    
            if t > 10:
                break
        
        if not int(ans):
            lcr.pattern_display('start')
    
    return image_LUT_entries_read, lut_read

def pattern_LUT_design(image_index_list, pat_number, exposure_period = 27084, frame_period = 33334):
    '''
    This function is used to create look up table and project the sequence for the projector based on the image_index_list (sequence) given. 

    :param image_index_list : The sequence to be created.
    :param pat_num : Pattern Number Mapping based on Table 2-70 in the programmer's guide.
    :param bit_depth : Desired bit-depth
    :param exposure_period: Exposure time in microseconds (4 bytes)
    :param frame_period: Frame period in microseconds (4 bytes).
    '''
    
    image_LUT_entries, swap_location_list = get_image_LUT_swap_location(image_index_list)
    
    proj_timing_start = perf_counter_ns()
    with connect_usb() as lcr:
        
        lcr.pattern_display('stop')
    
        # Get current status 
        lcr.get_main_status(pretty_print=True)
        #mode
        lcr.read_mode()
        lcr.retrieve_flashimages()
        lcr.read_pattern_input_source()
        lcr.set_pattern_config(num_lut_entries= len(image_index_list),
                              do_repeat = False,  
                              num_pats_for_trig_out2 = len(image_index_list),
                              num_images = len(image_LUT_entries))
        lcr.read_pattern_config()
        lcr.read_pattern_trigger_mode()
        lcr.read_trig_out1_control()
        lcr.set_exposure_frame_period(exposure_period  , frame_period )
        lcr.read_exposure_frame_period()
        # To set new image LUT
        lcr.open_mailbox(1)
        for i,j in enumerate(image_LUT_entries):
            lcr.mailbox_set_address(address = i)
            lcr.pattern_flash_index(j)
    
        lcr.mailbox_set_address(address = 0)
        image_LUT_entries_read = lcr.read_mailbox_info().tolist()[0:len(image_LUT_entries)]
        lcr.open_mailbox(0)
        
        #internal trigger
        trig_type = 0
        bit_depth = 8
        j = 0
       
        # To create new LUT
        lcr.open_mailbox(2)        
        for i in range(len(image_index_list)):
            
            lcr.mailbox_set_address(address = i)
            
            if i in swap_location_list:
                buffer_swap = True
            else:
                buffer_swap = False
            
            lcr.send_pattern_lut(trig_type = trig_type, pat_num = i % 3, bit_depth = bit_depth , led_select = 0b111, do_buf_swap = buffer_swap)            

        lcr.open_mailbox(0)
        
        # For reading newly created LUT table
        lcr.open_mailbox(2)
        lcr.mailbox_set_address(address = 0)
        temp = 3*len(image_index_list)
        lut_read = lcr.read_mailbox_info().tolist()[0:temp]
        lcr.open_mailbox(0)
        #start validation
        
        lcr.start_pattern_lut_validate()
        #Check validation status
        ret = 1
        start = perf_counter_ns()
        while ret:
            ans = lcr.read_lut_validate()
            ret = int(ans[0])    
            end = perf_counter_ns()
            t = (end - start)/1e9   
            if t > 10:
                print(t)
                break
           
        lcr.pattern_display('start')
    proj_timing_end = perf_counter_ns() 
    proj_timing = (proj_timing_end - proj_timing_start)/1e9    
    print('Projector Timing:%.3f sec'%proj_timing)
    image_index_list_recovered, swap_location_list_read = new_LUT_validation(image_index_list, swap_location_list, image_LUT_entries_read, lut_read)
    
    
    return image_LUT_entries_read, lut_read, image_index_list_recovered, swap_location_list, swap_location_list_read

def proj_cam_acquire_images(cam, acquisition_index, savedir, cam_triggerType, image_index_list, pat_number, proj_exposure_period, proj_frame_period):
    """
    This function acquires and saves one image from a device. Note that camera 
    must be initialized before calling this function, i.e., cam.Init() must be 
    called before calling this function.

    :param cam: Camera to acquire images from.
    :param savedir: directory to save images
    :param acquisition_index: the index number of the current acquisition.
    :param cam_triggerType: camera trigger type, must be one of {"software", "hardware"}
    :param image_index_list: projector pattern sequence to create and project
    :param proj_exposure_period : projector exposure period in microseconds 
    :param proj_frame_period : projector frame period in microseconds 
    :type cam: CameraPtr
    :type savedir: str
    :type acquisition_index: int
    :type triggerType: str
    :type: image_index_list: projector pattern sequence list
    :type: proj_exposure_period: float
    type: proj_frame_period: float
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** IMAGE ACQUISITION ***\n')

    result = True        

    # live view        
    cam.BeginAcquisition()        
    while True:                
        ret, frame = fa.capture_image(cam)       
        img_show = cv2.resize(frame, None, fx=0.5, fy=0.5)
        cv2.imshow("press q to quit", img_show)    
        key = cv2.waitKey(1)        
        if key == ord("q"):
            break
    cam.EndAcquisition()
    cv2.destroyAllWindows()
    
    # Retrieve, convert, and save image
    fa.activate_trigger(cam)
    cam.BeginAcquisition()        
    
    if cam_triggerType == "software":
        start = perf_counter_ns()            
        cam.TriggerSoftware.Execute()    
        ret, image_array = fa.capture_image(cam=cam)                
        end = perf_counter_ns()
        t = (end - start)/1e9
        print('time spent: %2.3f s' % t)                
        if ret:
            filename = 'Acquisition-%02d.jpg' %acquisition_index
            save_path = os.path.join(savedir, filename)                    
            cv2.imwrite(save_path, image_array)
            print('Image saved at %s' % save_path)
        else:
            print('Capture failed')
    
    if cam_triggerType == "hardware":
        count = 0        
        total_dual_time_start = perf_counter_ns()
        start = perf_counter_ns()   
        pattern_LUT_design(image_index_list, pat_number = pat_number , exposure_period = proj_exposure_period , frame_period = proj_frame_period )                    
        while count < len(image_index_list):
            try:
                ret, image_array = fa.capture_image(cam=cam)
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                ret = False
                image_array = None
                pass
                                
            if ret:
                print("extract successfully")
                filename = 'Acquisition-%02d-%02d.jpg' %(acquisition_index,count)
                save_path = os.path.join(savedir, filename)
                cv2.imwrite(save_path, image_array)
                print('Image saved at %s' % save_path)
                count += 1
                start = perf_counter_ns()
                print('waiting clock is reset')
            else:
                end = perf_counter_ns()
                waiting_time = (end - start)/1e9
                print('Capture failed, time spent %2.3f s before 10s timeout'%waiting_time)
                if waiting_time > 10:
                    print('timeout is reached, stop capturing image ...')
                    break
        total_dual_time_end = perf_counter_ns()
        total_dual_time = (total_dual_time_end - total_dual_time_start)/1e9
        print('Total dual device time:%.3f'%total_dual_time)
    
    cam.EndAcquisition()
    fa.deactivate_trigger(cam)        

    return result

def run_proj_single_camera(cam, savedir, acquisition_index, cam_triggerType, image_index_list, pat_number, proj_exposure_period, proj_frame_period ):
    """
    Initialize and configurate a camera and take one image.

    :param cam: Camera to run on.
    :type cam: CameraPtr
    :param savedir: directory to save images
    :param acquisition_index: the index of acquisition
    :param cam_triggerType: camera trigger type, must be one of {"software", "hardware"}
    :param image_index_list: projector pattern sequence to create and project
    :param proj_exposure_period : projector exposure period in microseconds 
    :param proj_frame_period : projector frame period in microseconds 
    :type savedir: str
    :type acquisition_index: int
    :type triggerType: str
    :type: image_index_list: projector pattern sequence list
    :type: proj_exposure_period: float
    type: proj_frame_period: float
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True
        # Initialize camera
        cam.Init()
        # config camera
        result &= fa.cam_configuration(cam, cam_triggerType)        
        # Acquire images        
        result &= proj_cam_acquire_images(cam, acquisition_index, savedir, cam_triggerType, image_index_list, pat_number, proj_exposure_period, proj_frame_period)
        # Deinitialize camera        
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False
    return result


#%%
# current_image_LUT_entries_read, current_lut_read = current_setting()

# #%%
# # image_index_list = [1,1,2,2,2,2,2]#,1,1,2,2,2]
# #image_index_list = [3,3,3,1,0,1,1,1,1,2,2,2,4,4,4]
# image_index_list = [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]
# gamma_index = np.repeat(np.arange(5,22),3).tolist()
# pat_number = [0,1,2]
# image_LUT_entries_read, lut_read, image_index_list_recovered, swap_location_list, swap_location_list_read = pattern_LUT_design(image_index_list, pat_number)
# #pattern_LUT_design(image_index_list)
# #%%

# proj_exposure_period = 27084
# proj_frame_period = 33334
# cam_triggerType = "hardware"
# result, system, cam_list, num_cameras = fa.sysScan()
# if result:
#     # Run example on each camera
#     savedir = r'C:\Users\kl001\Documents\grasshopper3_python\images'
#     fa.clearDir(savedir)
#     for i, cam in enumerate(cam_list):    
#         print('Running example for camera %d...'%i)
#         acquisition_index=0
#         result &= run_proj_single_camera(cam, savedir, acquisition_index, cam_triggerType, gamma_index,pat_number, proj_exposure_period, proj_frame_period )
#         print('Camera %d example complete...'%i)

#     # Release reference to camera
#     # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
#     # cleaned up when going out of scope.
#     # The usage of del is preferred to assigning the variable to None.
#     if cam_list:    
#         del cam
#     else:
#         print('Camera list is empty! No camera is detected, please check camera connection.')    
# else:
#     pass
# # Clear camera list before releasing system
# cam_list.Clear()
# # Release system instance
# system.ReleaseInstance() 

#%%

# with connect_usb() as lcr:
    
#     lcr.command('w', 0x00, 0x1a, 0x3A,)
#     lcr.pattern_display('start')





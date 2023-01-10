# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:27:49 2022

@author: kl001
"""
#TODO: Assemble files and set root directory
import numpy as np
import os
import sys
import time
from contextlib import contextmanager
import usb.core
from usb.core import USBError
from time import perf_counter_ns
import nstep_fringe as nstep
import cv2



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
    period = int(np.floor(1.0 / fps * 10**6))
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
    Class representing dmd controller. Can connect to different DLPCs by changing product ID. 
    Check IDs in device manager.
    """
    #TODO: Writing log file
    
    
    def __init__(self, device):
        """
        Connects the device.

        :param device: lcr4500 USB device.
        """
        #Initialse device address
        self.dlpc = device
        # Initialise properties of class (current status)
        self.get_main_status()
        self.retrieve_flashimages()
        self.read_mode()
        self.read_pattern_input_source()
        self.read_exposure_frame_period()
        self.read_pattern_config()
        self.read_pattern_trigger_mode()
        self.read_trig_out1_control()
        self.read_mailbox_info()        
        
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
            
    def pretty_print_status(self):
        print('\n====================================Current status of projector attributes=============================\n')
        print('\nMirror :{}'.format(self.mirrorStatus))
        print('\nSequencer :{}'.format(self.sequencer_status))
        print('\nBuffer:{}'.format(self.frame_buffer_status))
        print('\nGamma correction:{}'.format(self.gamma_correction))
        print('\nNo. of images in the flash:{}'.format(self.images_on_flash))
        print('\nMode:{}'.format(self.mode))
        print('\nSource:{} '.format(self.source))
        print('\nExposure period:{}'.format(self.exposure_period))
        print('\nFrame period:{}'.format(self.frame_period))
        print('\nNumber of LUT entries:{}\n \nIs pattern repeated:{}\n'.format(self.num_lut_entries,self.do_pattern_repeat,)) 
        print('\nNumber of patterns to display:{}\n \nNumber of images:{}'.format(self.num_pats_for_trig_out2, self.num_images))
        print('\nTrigger mode:{}'.format(self.trigger_mode))
        print('\nTrigger polarity:{}\n'.format(self.trigger_polarity,)) 
        print('\nTrigger rising edge delay:{} μs\n \nTrigger falling edge delay:{} μs'.format(self.trigedge_rise_delay_microsec,
                                                                                        self.trigedge_fall_delay_microsec))
        print('\nImage LUT entries:{}'.format(self.image_LUT_entries))
        print('\nPattern LUT entries:{}'.format(self.pattern_LUT_entries))
        
    def get_main_status(self):
        """The Main Status command shows the status of DMD park and DLPC350 sequencer, frame buffer, and gamma
         correction.

         (USB: CMD2: 0x02, CMD3: 0x0C)
         """
        self.command('r', 0x00, 0x1a, 0x0c, [])  # rw_mode, sequence,com1,com2=0x0c for main status, data
        ans = format(self.ans[4], '08b')
        
        if int(ans[-1]):
            self.mirrorStatus = 'parked'
        else:
            self.mirrorStatus = "not parked"
        if int(ans[-2]):
            self.sequencer_status = "running normally"
        else:
            self.sequencer_status = "stopped"
        if int(ans[-3]):
            self.frame_buffer_status = "frozen"
        else:
            self.frame_buffer_status = "not frozen"
        if int(ans[-4]):
            self.gamma_correction = "enabled"
        else:
            self.gamma_correction = "disabled"        
    
    def retrieve_flashimages(self):
        """
        This command retrieves the information about the number of Images in the flash.
        
        """
        
        self.command('r', 0x00, 0x1a, 0x42,[])
        self.images_on_flash = self.ans[4]
       
    def read_mode(self):  #default mode is 'pattern'
        """
        Read the current  input mode for the projector.

        (USB: CMD2: 0x1A, CMD3: 0x1B)
        """
        self.command('r', 0x00, 0x1a, 0x1b, [])
        ans = bin(self.ans[4]) 
        if int(ans[-1]):
            self.mode = "pattern"
        else:
            self.mode = "video"
        
    def set_display_mode(self, mode):  #default mode is 'pattern'
        """
        Selects the input mode for the projector.

        (USB: CMD2: 0x1A, CMD3: 0x1B)

        :param int mode:
            :0: "video" mode
            :1: "pattern" mode
        """
        self.mode = mode
        modes = ['video', 'pattern'] #video = 0, pattern =1
        if mode in modes:
            mode = modes.index(mode)

        self.command('w', 0x00, 0x1a, 0x1b, [mode])
        
    def read_pattern_input_source(self):
        """
        Read current input source.
        """
        self.command('r', 0x00, 0x1a, 0x22, [])
        ans = self.ans[4]
        if ans == 3:
            self.source = "flash"
        else:
            self.source = "video"
        
    def set_pattern_input_source(self, source='flash'):  # pattern source default = 'video'
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
        self.num_lut_entries = int(ans[-4]) + 1
        if int(ans[-3]):
            self.do_pattern_repeat = "yes"
        else:
            self.do_pattern_repeat = "no"
        self.num_pats_for_trig_out2 =  int(ans[-2])+1
        self.num_images = int(ans[-1])+1
        
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
        self.num_lut_entries = num_lut_entries
        self.do_pattern_repeat = do_repeat
        self.num_pats_for_trig_out2 = num_pats_for_trig_out2
        self.num_images = num_images
        
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
        self.trigger_mode = ans
        
    def set_pattern_trigger_mode(self, trigger_mode='vsync'):
        """
        Selects the trigger type for pattern sequence.

        (USB: CMD2: 0x1A, CMD3: 0x23)

        :param int mode:
            :0: "vsync"
        """
        self.trigger_mode = trigger_mode
        trigger_modes = ['vsync','trig_mode1', 'trig_mode2', 'trig_mode3', 'trig_mode4']
        if trigger_mode in trigger_modes:
            trigger_mode = trigger_modes.index(trigger_mode)

        self.command('w', 0x00, 0x1a, 0x23, [trigger_mode])
    
    def read_trig_out1_control(self):
        """
        Read current trig_out1 setting.
        """
        self.command('r', 0x00, 0x1a, 0x1d, [])
        ans = self.ans[4:7]
        if int(ans[0]):
            self.trigger_polarity = "active low signal"
        else:
            self.trigger_polarity = "active high signal"
            
        tigger_rising_edge_delay = ans[1]
        trigger_falling_edge_delay = ans[-1]
        #convert to μs
        self.trigedge_rise_delay_microsec = np.round(-20.05 + 0.1072 * tigger_rising_edge_delay, decimals = 3)
        self.trigedge_fall_delay_microsec = np.round(-20.05 + 0.1072 * trigger_falling_edge_delay, decimals = 3)
        
    def trig_out1_control(self,polarity_invert = True, trigedge_rise_delay_microsec = 0, trigedge_fall_delay_microsec = 0):
         """
         The Trigger Out1 Control command sets the polarity, rising edge delay, 
         and falling edge delay of the TRIG_OUT_1 signal of the DLPC350. 
         Before executing this command, stop the current pattern sequence. After executing this command, 
         send the Validation command (I2C: 0x7D or USB: 0x1A1A) once before starting the pattern sequence.
         
         param bool plarity_invert: True for active low signal
         param int trigedge_rise_delay: rising edge delay control ranging from –20.05 μs to 2.787 μs. Each bit adds 107.2 ns.
         param int trigedge_fall_delay: falling edge delay control with range -20.05 μs to +2.787 μs. Each bit adds 107.2 ns
         """
         #convert μs to number equivalent
         trigedge_rise_delay = int(trigedge_rise_delay_microsec - (-20.05))/0.1072
         trigedge_fall_delay = int(trigedge_fall_delay_microsec - (-20.05))/0.1072
         if polarity_invert:
             polarity = '00000010'
             self.trigger_polarity = "active low signal"
         else:
             polarity = '00000000'
             self.trigger_polarity = "active high signal"
             
         self.trigedge_rise_delay_microsec = trigedge_rise_delay_microsec
         self.trigedge_fall_delay_microsec = trigedge_fall_delay_microsec
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
        self.exposure_period = ans[4] + ans[5]*256 + ans[6]*256**2 + ans[7]*256**3
        self.frame_period = ans[8] + ans[9]*256 + ans[10]*256**2 + ans[11]*256**3
    
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
         
         ret = 1
         start = perf_counter_ns()
         while ret:
             self.command('r', 0x00, 0x1a, 0x1a,[])
             ans = conv_len(self.ans[4],8)
             ret = int(ans[0])    
             end = perf_counter_ns()
             t = (end - start)/1e9    
             if t > 10:
                 break
         print('\n================= Validation result ======================\n')
         print(f'Exposure and frame period setting: {"invalid" if int(ans[-1]) else "valid"}\n')
         print(f'LUT: {"invalid" if int(ans[-2]) else "valid"}\n')
         print(f'Trigger Out1: {"invalid" if int(ans[-3]) else "valid"}\n')
         print(f'Post sector settings: {"warning:invalid" if int(ans[-4]) else "valid"}\n')
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
        
    def image_flash_index(self, index_list, address):
        """
        The following parameters: display mode, trigger mode, exposure, 
        and frame rate must be set up before sending any mailbox data.
        If the mailbox is opened to define the flash image indexes, list the index numbers in the mailbox. 
        For example, if image indexes 0 through 3 are desired, write 0x0 0x1 0x2 0x3 to the mailbox. 
        
        :param index_list: image index list to be written
        """
        self.open_mailbox(1)
        self.mailbox_set_address(address = address)      
        self.command('w', 0x00, 0x1a, 0x34, index_list)
        self.open_mailbox(0)
        
    def pattern_lut_payload_list(self,trig_type,
                                 bit_depth,
                                    led_select, swap_location_list, 
                                    image_index_list,
                                    pattern_num_list,
                                    do_invert_pat=False,
                                    do_insert_black=True,
                                    do_trig_out_prev=False):
        '''
        Function to create payload for pattern LUT
        '''
        payload_list = []
        trig_type = conv_len(trig_type, 2)
        bit_depth = conv_len(bit_depth, 4)
        led_select = conv_len(led_select, 4)
        # byte 1
        byte_1 = led_select + bit_depth
        # byte 2
        do_invert_pat = str(int(do_invert_pat))
        do_insert_black = str(int(do_insert_black))
        do_trig_out_prev = str(int(do_trig_out_prev))
        
        if len(image_index_list) != len(pattern_num_list):
            print('ERROR: length of image list is not compatible with that of pattern number list')
            return None
        for i in range(len(image_index_list)):
            if i in swap_location_list:
                buffer_swap = True
            else:
                buffer_swap = False
            # pat_num = conv_len( i % 3, 6)
            byte_0 = conv_len(pattern_num_list[i],6) + trig_type
            do_buf_swap = str(int(buffer_swap))
            byte_2 = '0000' + do_trig_out_prev + do_buf_swap + do_insert_black + do_invert_pat
            payload = byte_2 + byte_1 + byte_0
            payload = bits_to_bytes(payload)
            payload_list.extend(payload)
        return payload_list
            
    def send_pattern_lut(self,
                         trig_type,
                         bit_depth,
                         led_select, 
                         swap_location_list, 
                         image_index_list, 
                         pattern_num_list, 
                         starting_address,
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
        self.open_mailbox(2) 
        payload_flat_list = self.pattern_lut_payload_list(trig_type,
                                                          bit_depth,
                                                          led_select, 
                                                          swap_location_list, 
                                                          image_index_list, 
                                                          pattern_num_list, 
                                                          do_invert_pat,
                                                          do_insert_black,
                                                          do_trig_out_prev)
        
        self.mailbox_set_address(address = starting_address)
        self.command('w', 0x00, 0x1a, 0x34, payload_flat_list)
        self.open_mailbox(0) 
        self.read_mailbox_info() # to update the image and pattern LUT table
        
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
        '''
        This function reads image LUT table and pattern LUT contents
        :param bool read_image_index: Read image table True or False
        '''
        #Read image table
        self.open_mailbox(1)
        self.mailbox_set_address(address = 0)
        self.command('r', 0x00, 0x1a, 0x34, [])
        ans =self.ans[4:].tolist()
        self.image_LUT_entries = ans
        self.open_mailbox(0)
        
        #Read pattern LUT
        self.open_mailbox(2)
        self.mailbox_set_address(address = 0)
        self.command('r', 0x00, 0x1a, 0x34, [])
        ans =self.ans[4:].tolist()
        self.pattern_LUT_entries = ans
        self.open_mailbox(0)
        
def get_image_LUT_swap_location(image_index_list):
    swap_location_list = [0] + [i for i in range(1,len(image_index_list)) if image_index_list[i]!=image_index_list[i-1]]
    image_LUT_entries = [image_index_list[i] for i in swap_location_list]

    if len(image_LUT_entries) == 2:    
        temp = image_LUT_entries[0:2].copy()
        image_LUT_entries[0] = temp[1]
        image_LUT_entries[1] = temp[0]
        
    return image_LUT_entries, swap_location_list

def LUT_verification(image_index_list, swap_location_list,  image_LUT_entries_read, lut_read):
    '''
    This function compares the projector LUT with the user requested LUT to verify the correct LUT is generated in the projector.
    The projector has two LUTs, one with only image index order, and other with image bit plane order.
    :param list image_index_list: user requested LUT
    :param list swap_location_list: locations of buffer swap according to user image_index_list
    :param list image_LUT_entries_read: projector image index read from projector
    :param list lut_read: projector LUT read from projector.
    '''      
    
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

    num_repeats =  [swap_location_list_read[i] - swap_location_list_read[i-1] for i in range(1,len(swap_location_list_read))] + [num_of_patterns - swap_location_list_read[-1]]
    image_index_list_recovered = []
    for i,num in enumerate(num_repeats):
        image_index_list_recovered.extend([image_LUT_entries_temp[i]]*num)
        
    if image_index_list == image_index_list_recovered:
        print("\nRecovery successfull")
        result = True
    else:
        print("\nRecovery test failed")
        result = False
    
    return result

def current_setting():
    with connect_usb() as lcr:
        # to stop current pattern sequence mode
        lcr.pattern_display('stop')
        lcr.pretty_print_status()
        # do validation and project current LUT patterns
        ans = lcr.start_pattern_lut_validate()
        if not int(ans):
            lcr.pattern_display('start')
    return 

def forge_bmp(single_channel_image_list, savedir, convertRGB = True):
    '''
    Function to create list of 3 channel (24 bit) image from list of single channel (8 bit) images.
    :param single_channel_image_list : list of 8 bit images
    :param convertRGB: If set each image will be RGB otherwise BGR
    '''
        
    image_array = np.empty((single_channel_image_list[0].shape[0],single_channel_image_list[0].shape[1],3))
    three_channel_list = []
    count = 0
    for j,i in enumerate(single_channel_image_list):
        image_array[:,:,(j%3)] = i
        if j%3 == 2: #0,1,2
            if  convertRGB:
                image_array[:,:,[0,1,2]] = image_array[:,:,[2,0,1]] # changing channel for saving for projector (RGB)
            cv2.imwrite(os.path.join(savedir,"image_%d.bmp"%count), image_array)
            three_channel_list.append(image_array)
            image_array = np.empty((single_channel_image_list[0].shape[0],single_channel_image_list[0].shape[1],3))
            count+=1
            
    if len(single_channel_image_list)%3 != 0 :
        print("Warning: Last image in the list is a %d channel image"%(len(single_channel_image_list)%3))
            
    return three_channel_list 

def forge_fringe_bmp(savedir, 
                       pitch_list, 
                       N_list, 
                       type_unwrap, 
                       phase_st, 
                       inte_rang, direc = 'v', 
                       calib_fringes = False, 
                       proj_width = 912, 
                       proj_height = 1140):
    
    if calib_fringes:
        fringe_array, delta_deck_list = nstep.calib_generate(proj_width, 
                                                             proj_height, 
                                                             type_unwrap, 
                                                             N_list, 
                                                             pitch_list, 
                                                             phase_st, 
                                                             inte_rang, 
                                                             savedir)
    else:
        fringe_array, delta_deck_list = nstep.recon_generate(proj_width, 
                                                             proj_height, 
                                                             type_unwrap, 
                                                             N_list, 
                                                             pitch_list, 
                                                             phase_st, 
                                                             inte_rang, 
                                                             direc, 
                                                             savedir)
    
    fringe_bmp_list = forge_bmp(fringe_array, savedir, convertRGB = True)
    
    return fringe_bmp_list

def proj_single_img(image_index,
                    exposure_period = 27084, 
                    frame_period = 33334):
    
    '''
    Function projects single image with given flash image index in repeat mode.
    :param int image_index: Index of image on projector flash
    '''
  
    with connect_usb() as lcr:
        try:
            result = True
            lcr.pattern_display('stop')
            lcr.set_pattern_config(num_lut_entries= 1, 
                                   do_repeat = True, 
                                   num_pats_for_trig_out2 = 1, 
                                   num_images = 1)
            lcr.set_exposure_frame_period(exposure_period, frame_period )
            
            lcr.image_flash_index([image_index],0)
            
            lcr.send_pattern_lut(trig_type = 0,
                                bit_depth = 8 ,
                                led_select = 0b111, 
                                swap_location_list = [0], 
                                image_index_list = [image_index], 
                                pattern_num_list = [0], 
                                starting_address = 0,
                                do_invert_pat=False,
                                do_insert_black=False,
                                do_trig_out_prev=False)
                               
            lcr.pretty_print_status()
            ans = lcr.start_pattern_lut_validate()
#TODO: Check on Post sector settings. It is only a warning in gui for this exposure and frame period. Display still works.
            if (not int(ans)) or (int(ans,2)==8):
                lcr.pattern_display('start')
                
                test_image = np.full((1140,912),255, dtype=np.uint8)
                while True:
                    cv2.imshow("press q to quit", test_image)  
                    key = cv2.waitKey(1)    
                    if key == ord("q"):
                        lcr.pattern_display('stop')
                        break
                cv2.destroyAllWindows()
            else:
                result &=False
        except:
            print("An exception occurred")
            result = False
        
    return result

def proj_pattern_LUT(image_index_list, 
                       pattern_num_list, 
                       exposure_period = 27084, 
                       frame_period = 33334, 
                       pprint_proj_status = True ):
    '''
    This function is used to create look up table and project the sequence for the projector based on the 
    image_index_list (sequence) given. 

    :param image_index_list : The sequence to be created.
    :param bit_depth : Desired bit-depth
    :param exposure_period: Exposure time in microseconds (4 bytes)
    :param frame_period: Frame period in microseconds (4 bytes).
    :pprint_proj_status: If set will print projector's current status.
    '''
    
    image_LUT_entries, swap_location_list = get_image_LUT_swap_location(image_index_list)
    
    proj_timing_start = perf_counter_ns()
    with connect_usb() as lcr:
        try:
            result = True
            lcr.pattern_display('stop')
        
            lcr.set_pattern_config(num_lut_entries= len(image_index_list),
                                  do_repeat = False,  
                                  num_pats_for_trig_out2 = len(image_index_list),
                                  num_images = len(image_LUT_entries))
            lcr.set_exposure_frame_period(exposure_period, frame_period )
            # To set new image LUT
            lcr.image_flash_index(image_LUT_entries,0)
            # To set pattern LUT table    
            lcr.send_pattern_lut(trig_type = 0, 
                                 bit_depth = 8, 
                                 led_select = 0b111, 
                                 swap_location_list = swap_location_list, 
                                 image_index_list = image_index_list, 
                                 pattern_num_list = pattern_num_list, 
                                 starting_address = 0)
            if pprint_proj_status:# Print all projector current attributes set
                lcr.pretty_print_status()
            image_LUT_entries_read = lcr.image_LUT_entries[0:len(image_LUT_entries)]
            temp = 3*len(image_index_list)
            lut_read = lcr.pattern_LUT_entries[0:temp]
            #start validation
            ans = lcr.start_pattern_lut_validate()
            #Check validation status
            if (not int(ans)) or (int(ans,2)==8):   
                lcr.pattern_display('start')
        except:
            print("An exception occurred")
            result = False
    proj_timing_end = perf_counter_ns() 
    proj_timing = (proj_timing_end - proj_timing_start)/1e9    
    print('Projector Timing:%.3f sec'%proj_timing)
    result &= LUT_verification(image_index_list, 
                              swap_location_list, 
                              image_LUT_entries_read, 
                              lut_read)
    return result


#%%

def main():
    result = True
    result &= proj_single_img(image_index = 34)
    image_index_list = np.repeat(np.arange(0,5),3).tolist()
    pattern_num_list = [0,1,2] * len(set(image_index_list))
    result &= proj_pattern_LUT(image_index_list, 
                                pattern_num_list)
    return result

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)



# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:27:49 2022

@author: kl001
"""
# TODO: Assemble files and set root directory
#TODO: Time image loading. pattern period * no.of patterns per image > image load time
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
    Padding '0' before convert number to make convert number has same length with given length.
    :param a: Number to convert.
    :param l: Length of bit string.
    :type a : int
    :type l : int
    :return: Padded bit string.
    :rtype: str
    """
    b = bin(a)[2:]
    padding = l - len(b)
    b = '0' * padding + b
    return b


def bits_to_bytes(a, reverse=True):  # default is reverse
    """
    Function that converts bit string into a given number of bytes.
    First check if length less than 8, if not padding '0' before, then convert bits number into bytes. 
    Default reverse set as True (First in last out).
    :param a: Bytes to convert.
    :param reverse: Whether to reverse the byte list.
    :type a: str
    :type reverse: bool
    :return: List of bytes.
    """
    bytelist = []

    # check if it needs padding
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
    Calculates desired period (us) from given FPS.

    :param fps: Frames per second.
    :type fps: int
    :return period: Period (us).
    :rtype period: int
    """
    period = int(np.floor(1.0 / fps * 10**6))
    return period


@contextmanager
def connect_usb():
    """
    Context manager for connecting to and releasing usb device.
    For DLPC350, Product ID is 0x6401, and Vendor ID is 0x0451.
    :yields: USB device.
    """
    device = usb.core.find(idVendor=0x0451, idProduct=0x6401)  # finding the projector usb port
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
    # TODO: Writing log file
    def __init__(self, device):
        """
        Connects the device.
        :param device: lcr4500 USB device.
        :type device: ptr
        """

        # Initialise device address
        self.dlpc = device
        self.mirrorStatus = None
        self.sequencer_status = None
        self.frame_buffer_status = None
        self.gamma_correction = None
        self.images_on_flash = None
        self.trigger_polarity = None
        self.mode = None
        self.source = None
        self.num_lut_entries = None
        self.do_pattern_repeat = None
        self.num_pats_for_trig_out2 = None
        self.num_images = None
        self.trigger_mode = None
        self.trigedge_rise_delay_microsec = None
        self.trigedge_fall_delay_microsec = None
        self.exposure_period = None
        self.frame_period = None
        self.image_LUT_entries = None
        self.pattern_LUT_entries = None
        self.ans = None
        # Initialise properties of class (read default status)
        self.read_main_status()
        self.read_num_of_flashimages()
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
        #TODO: check and edit next 3 line
        Sends a command to the dlpc.The order of the command always be 
        {[read (120) or write (80) mode], [sequency], [CMD3], [CMD2], [Data]}. If there are second commend requested,
        the second one only contains {[Data]}.
        When read form projector, the first 4 space always be occupied by those order.
        From DLPC Programming guide:
        Byte0 report ID byte: User don't need to setup this parameter. Report ID = 0.
        Byte1 Flag byte: Bits 2:0 are set to 0x0 for regular DLPC350 operation, set 0x7 for debugging assistance.
                         Bit 6 set to 0x1 indicates host needs reply from device.
                         Bit 7 set to 0x1 indicates a read transaction, 0x0 as write. 
        Byte2 Sequence byte: A signal command contains no more than 64 bytes (contains up to 20 patterns or 6 images with RGB channel)
                             the command is sent as multiple USB packets and the sequence byte numbers the packets 
                             so the device can assemble them in the right sequence.
        Byte3 & Byte4: Length LSB and MSB. User don't need to setup this parameter, it be calculated in function. 
                       This length denotes the number of data bytes in the packet and excludes the number of bytes 0-4.
        Subcommand bytes: CMD2 and CMD3.
        Byte5 and beyond: Data byte. 
    
        :param str rw_mode: Whether reading or writing.
        :param sequence_byte:
        :param com1: Command 1
        :param com2: Command 2
        :param data: Data to pass with command.
        :type rw_mode: str
        :type sequence_byte: int
        :type com1: int
        :type com2: int
        :type data: list
        :return result: True if successful, False otherwise.
        :rtype result:bool
        """

        buffer = []
        result = True

        if rw_mode == 'r':
            flagstring = 0xc0  # 0b11000000
        else:
            flagstring = 0x40  # 0b01000000

        data_len = conv_len(len(data) + 2, 16)  # size of data + subcommands to 16 bits binary
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
        
        # listen to the response from the device for verification
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
            result = False

        time.sleep(0.02)
        return result

    def print_reply(self):
        """
        Print bytes in reply(Hex).
        """
        for i in self.ans:
            print(hex(i))
            
    def pretty_print_status(self):
        """
        Function to pretty print projector current attributes.
        """
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
        print('\nNumber of LUT entries:{}\n \nIs pattern repeated:{}\n'.format(self.num_lut_entries,
                                                                               self.do_pattern_repeat,))
        print('\nNumber of patterns to display:{}\n \nNumber of images:{}'.format(self.num_pats_for_trig_out2,
                                                                                  self.num_images))
        print('\nTrigger mode:{}'.format(self.trigger_mode))
        print('\nTrigger polarity:{}\n'.format(self.trigger_polarity,)) 
        print('\nTrigger rising edge delay:{} μs\n \nTrigger falling edge delay:{} μs'.format(self.trigedge_rise_delay_microsec,
                                                                                              self.trigedge_fall_delay_microsec))
        print('\nImage LUT entries:{}'.format(self.image_LUT_entries))
        print('\nPattern LUT entries:{}'.format(self.pattern_LUT_entries))
        
    def read_main_status(self):
        """
        The Main Status command shows the status of DMD park and DLPC350 sequencer, frame buffer, and gamma
        correction. If everything is working properly，receive 0b00001111.  
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x0c, [])  # rw_mode, sequence,com1=0x1a,com2=0x0c for main status, data
        if result:            
            ans = format(self.ans[4], '08b')  # convert int into 8bit binary, 08: paded '0' on left side.
            
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
        else:
            self.mirrorStatus = None            
            self.sequencer_status = None
            self.frame_buffer_status = None
            self.gamma_correction = None
        return result
    
    def read_num_of_flashimages(self):
        """
        This command retrieves the information about the number of Images in the flash.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        
        result = self.command('r', 0x00, 0x1a, 0x42, [])
        if result:
            self.images_on_flash = self.ans[4]
        else:
            self.images_on_flash = None
        return result
       
    def read_mode(self):  # default mode is 'pattern'
        """
        Read the current input mode for the projector.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x1b, [])
        if result:
            ans = bin(self.ans[4]) 
            if int(ans[-1]):
                self.mode = "pattern"
            else:
                self.mode = "video"
        else:
            self.mode = None
        return result
        
    def set_display_mode(self, mode):
        """
        Sets the Display/Operating mode(Video mode or Pattern mode) for the projector.         
        video mode: Work as the normal projector with pixel resolution up to 1280x800 up to 120Hz.
                    Gamma correction and input display resolution only support by video mode.
        Pattern mode: Read image from flash with pixel resolution up to 912x1140. 
        :param mode: 'video' or 'pattern'
        :type mode: str
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
       
        modes = ['video', 'pattern']  # video = 0, pattern =1
        if mode in modes:
            mode_no = modes.index(mode)
            result = self.command('w', 0x00, 0x1a, 0x1b, [mode_no])
        else:
            result = False
            print('ERROR: Required display mode is not supported. Mode must be one of {"video", "pattern"}')
        if result:
            self.mode = mode
        return result
        
    def read_pattern_input_source(self):
        """
        Read current pattern input source.
        This function support by selecting pattern mode.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x22, [])
        if result:
            ans = self.ans[4]
            if ans == 3:
                self.source = "flash"
            else:
                self.source = "video"
        else:
            self.source = None
        return result
        
    def set_pattern_input_source(self, source='flash'):  # pattern source default = 'flash'
        """
        Selects the input source for pattern sequence. This pattern source can be set read from 
        flash or video port. Before executing this command, stop the current pattern sequence. 
        After executing this command, send the validation command(USB:0x1A1A) once before starting 
        the pattern sequence.
        :param source: "video" ,"flash"
        :type source: str
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        sources = ['video', '', '', 'flash']  # video = 0, reserved, reserved, flash=11 (bin 3)
        if source in sources:
            source_no = sources.index(source)
            result = self.command('w', 0x00, 0x1a, 0x22, [source_no])
        else:
            result = False
            print('ERROR: Required pattern input source is not supported.')
        if result:
            self.source = source
        return result
    
    def read_pattern_config(self):
        """
        Read current pattern lookup table.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x31, [])
        if result:
            ans = self.ans[4:8]
            self.num_lut_entries = int(ans[-4]) + 1
            if int(ans[-3]):
                self.do_pattern_repeat = "yes"
            else:
                self.do_pattern_repeat = "no"
            self.num_pats_for_trig_out2 = int(ans[-2])+1
            self.num_images = int(ans[-1])+1
        else:
            self.num_lut_entries = None
            self.do_pattern_repeat = None
            self.num_pats_for_trig_out2 = None
            self.num_images = None
        return result
        
    def set_pattern_config(self,
                           num_lut_entries=15,
                           do_repeat=False,  # Default repeat pattern
                           num_pats_for_trig_out2=15,
                           num_images=5):
        """
        This API controls the execution of patterns stored in the lookup table. Before using this API, stop the current
        pattern sequence using ``DLPC350_PatternDisplay()`` API. After calling this API, send the Validation command
        using the API DLPC350_ValidatePatLutData() before starting the pattern sequence.
        When padding several bytes, byte0 on the right side.
        (USB: CMD2: 0x1A, CMD3: 0x31)
        :param num_lut_entries: Number of LUT entries(Range from 1 to 128).
        :param do_repeat:True: Execute the pattern sequence once. False: Repeat the pattern sequence.
        :param num_pats_for_trig_out2: Number of patterns to display(range 1 through 256). If in repeat mode, then
                                       this value dictates how often TRIG_OUT_2 is generated.
        :param num_images: Number of Image Index LUT Entries(range 1 through 64). This Field is irrelevant for Pattern
                           Display Data Input Source set to a value other than internal.
        :type num_lut_entries: int
        :type do_repeat: bool
        :type num_pats_for_trig_out2: int
        :type num_images: int
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        
        num_lut_entries_bin = '0' + conv_len(num_lut_entries - 1, 7)  # Byte0: 6:0 LUT, 7: Reserved
        do_repeat_bin = '0000000' + str(int(do_repeat))  # Byte1: 0 Repeat pattern seq, 7:1: Reserved
        num_pats_for_trig_out2_bin = conv_len(num_pats_for_trig_out2 - 1, 8)  # Byte2: 7:0 Pattern number
        num_images_bin = '00' + conv_len(num_images - 1, 6)  # Byte3: 5:0 Image index, 7:6 Reserved

        payload = num_images_bin + num_pats_for_trig_out2_bin + do_repeat_bin + num_lut_entries_bin
        payload = bits_to_bytes(payload)

        result = self.command('w', 0x00, 0x1a, 0x31, payload)
        if result:
            self.num_lut_entries = num_lut_entries
            self.do_pattern_repeat = do_repeat
            self.num_pats_for_trig_out2 = num_pats_for_trig_out2
            self.num_images = num_images
        return result
    
    def read_pattern_trigger_mode(self):
        """
        Read current trigger mode.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x23, [])
        if result:
            ans = self.ans[4]
            self.trigger_mode = ans
        else:
            self.trigger_mode = None
        return result
        
    def set_pattern_trigger_mode(self, trigger_mode='vsync'):
        """
        Selects the trigger mode for pattern sequence.
        Pattern Trigger Mode 0: VSYNC triggers the pattern display sequence.For proper operation,
                                the pattern exposure must equal the total pattern period in this mode.
        Pattern Trigger Mode 1: Internally or externally (through TRIG_IN_1 and TRIG_IN_2) generated trigger.
        Pattern Trigger Mode 2: TRIG_IN_1 alternates between two patterns and TRIG_IN_2 advances to the next pair of patterns.
        Pattern Trigger Mode 3: Internally or externally generated trigger for variable exposure display sequence.
        Pattern Trigger Mode 4: VSYNC triggered for variable exposure display sequence.Exposure must equal the total
                                pattern period in this mode.
        :param trigger_mode :0: 'vsync'
                            :1: 'trig_mode1'
                            :2: 'trig_mode2'
                            :3: 'trig_mode3'
                            :4: 'trig_mode4'
        :type trigger_mode: str
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        trigger_modes = ['vsync', 'trig_mode1', 'trig_mode2', 'trig_mode3', 'trig_mode4']
        if trigger_mode in trigger_modes:
            trigger_mode_no = trigger_modes.index(trigger_mode)
            result = self.command('w', 0x00, 0x1a, 0x23, [trigger_mode_no])
        else:
            print("ERROR: Requested trigger mode is not supported, trigger mode must be one of {'vsync', 'trig_mode1', 'trig_mode2', 'trig_mode3', 'trig_mode4'}")
            result = False
        if result:
            self.trigger_mode = trigger_mode
        return result
    
    def read_trig_out1_control(self):
        """
        Read current trig_out1 setting.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x1d, [])
        if result:
            ans = self.ans[4:7]
            if int(ans[0]):
                self.trigger_polarity = "active low signal"
            else:
                self.trigger_polarity = "active high signal"
                
            tigger_rising_edge_delay = ans[1]  # Receive a hex number range from 0 to 213, 187 with delay 0us.
            trigger_falling_edge_delay = ans[-1]
            # convert to μs.
            self.trigedge_rise_delay_microsec = np.round(-20.05 + 0.1072 * tigger_rising_edge_delay, decimals=2)
            self.trigedge_fall_delay_microsec = np.round(-20.05 + 0.1072 * trigger_falling_edge_delay, decimals=2)
        else:
            self.trigger_polarity = None
            self.trigedge_rise_delay_microsec = None
            self.trigedge_fall_delay_microsec = None
        return result
        
    def trig_out1_control(self,
                          polarity_invert=True,
                          trigedge_rise_delay_microsec=0,
                          trigedge_fall_delay_microsec=0):
        """
        The Trigger Out1 Control command sets the polarity, rising edge delay, 
        and falling edge delay of the TRIG_OUT_1 signal of the DLPC350. 
        Before executing this command, stop the current pattern sequence. After executing this command, 
        send the Validation command (I2C: 0x7D or USB: 0x1A1A) once before starting the pattern sequence.
        Normal polarity: high signal.
        :param polarity_invert: True for active low signal
        :param trigedge_rise_delay_microsec: rising edge delay control ranging from –20.05 μs to 2.787 μs. Each bit adds 107.2 ns.
        :param trigedge_fall_delay_microsec: falling edge delay control with range -20.05 μs to +2.787 μs. Each bit adds 107.2 ns
        :type polarity_invert: bool
        :type trigedge_rise_delay_microsec: int
        :type trigedge_fall_delay_microsec: int
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        # convert μs to number equivalent
        trigedge_rise_delay = int((trigedge_rise_delay_microsec - (-20.05))/0.1072)
        trigedge_fall_delay = int((trigedge_fall_delay_microsec - (-20.05))/0.1072)
        if polarity_invert:
            polarity = '00000010'  # Bit0: reserved. Bit1: 1: active low signal 0: active high signal. Bit 7:2: Reserved
            self.trigger_polarity = "active low signal"
        else:
            polarity = '00000000'
            self.trigger_polarity = "active high signal"
        trigedge_rise_delay = conv_len(trigedge_rise_delay, 8)
        trigedge_fall_delay = conv_len(trigedge_fall_delay, 8)
        payload = trigedge_fall_delay + trigedge_rise_delay + polarity
        payload = bits_to_bytes(payload)
        result = self.command('w', 0x00, 0x1a, 0x1d, payload)
        if result:
            self.trigedge_rise_delay_microsec = trigedge_rise_delay_microsec
            self.trigedge_fall_delay_microsec = trigedge_fall_delay_microsec
        return result

    def read_exposure_frame_period(self):
        """
        Read current exposure time and frame period.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x29, [])
        if result:
            ans = self.ans
            self.exposure_period = ans[4] + ans[5]*256 + ans[6]*256**2 + ans[7]*256**3
            self.frame_period = ans[8] + ans[9]*256 + ans[10]*256**2 + ans[11]*256**3
        else:
            self.exposure_period = None
            self.frame_period = None
        return result
    
    def set_exposure_frame_period(self, 
                                  exposure_period, 
                                  frame_period):
        """
        The Pattern Display Exposure and Frame Period dictates the time a pattern is exposed and the frame period.
        Either the exposure time must be equivalent to the frame period, or the exposure time must be less than the
        frame period by 230 microseconds. Before executing this command, stop the current pattern sequence. After
        executing this command, call ``DLPC350_ValidatePatLutData()`` API before starting the pattern sequence.
        Byte3:0 bit31:0: Pattern exposure time (μs). Dicitates how long the display time is. Since the exposure has 
                         at least 230 μs difference, during the difference, projector stop projecting.
        Byte7:4 bit31:0: Frame period (μs). Dicitates the interval between 2 frames.
        :param exposure_period: Exposure time in microseconds (4 bytes).
        :param frame_period: Frame period in microseconds (4 bytes).
        :type exposure_period: int
        :type frame_period: int
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        
        exposure_period_bin = conv_len(exposure_period, 32)  # decimal to bit string of size 32
        frame_period_bin = conv_len(frame_period, 32)  # decimal to bit string of size 32

        payload = frame_period_bin + exposure_period_bin
        payload = bits_to_bytes(payload)  # it will be reverse

        result = self.command('w', 0x00, 0x1a, 0x29, payload)
        if result:
            self.exposure_period = exposure_period
            self.frame_period = frame_period
        return result
  
    def start_pattern_lut_validate(self):
        """
        This API checks the programmed pattern display modes and indicates any invalid settings. This command needs to
        be executed after all pattern display configurations have been completed.
        Several setting validate in this function including:
            Bit0: Exposure or frame period setting.
            Bit1: Patten number in LUT.
            Bit2: Tigger out1 setting.
            Bit3: Post sector setting.
            Bit4: Frame period and exposure differece.
            Bit6 & Bit5: Reserved.
            Bit7: Status of DLPC350 validating.
        First make every bit to invalid(0b11111111) which gives enough time(10 sec) for validation,
        if validation time larger than 10 sec, stop running.
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('w', 0x00, 0x1a, 0x1a, bits_to_bytes(conv_len(0x00, 8)))  # Pattern Display Mode: Validate Data: CMD2: 0x1A, CMD3: 0x1A
        if result:            
            ans = '11111111'
            ret = int(ans[0], 2)
            start = perf_counter_ns()
            while ret:
                result &= self.command('r', 0x00, 0x1a, 0x1a, [])
                if result:
                    ans = conv_len(self.ans[4], 8)
                    ret = int(ans[0], 2)
                end = perf_counter_ns()
                t = (end - start)/1e9
                if t > 10:
                    result &= False
                    print('\n Validation time longer than 10s \n')
                    break

            if int(ans):
                if ret == 255:
                    print('\n ERROR: validation timeout. Reason: Cannot hear back from projector, ')
                else:
                    print('\n ERROR: Validation failure, see results below\n ')
                result &= False
            else:
                print('\nValidation successful \n')
                result &= True
            
            if ret != 255:
                print('\n================= Validation result ======================\n')
                print(f'Exposure and frame period setting: {"invalid" if int(ans[-1]) else "valid"}\n')
                print(f'LUT: {"invalid" if int(ans[-2]) else "valid"}\n')
                print(f'Trigger Out1: {"invalid" if int(ans[-3]) else "valid"}\n')
                print(f'Post sector settings: {"warning:invalid" if int(ans[-4]) else "valid"}\n')
                print(f'DLPC350 is {"busy" if int(ans[-8]) else "valid"}\n')

            return result
        else:
            print('ERROR: Cannot execute validate operation.')
            return result
        
    def open_mailbox(self, mbox_num):
        """
        This API opens the specified Mailbox within the DLPC350 controller. This API must be called before sending data
        to the mailbox/LUT using DLPC350_SendPatLut() or DLPC350_SendImageLut() APIs.
        Opening and closing maibox must occur in pairs.
        :param mbox_num :0: Disable (close) the mailboxes.
                        :1: Open the mailbox for image index configuration.
                        :2: Open the mailbox for pattern definition.
                        :3: Open the mailbox for the Variable Exposure.
        :type mbox_num:int
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        mbox_num = bits_to_bytes(conv_len(mbox_num, 8))
        result = self.command('w', 0x00, 0x1a, 0x33, mbox_num)
        return result
        
    def mailbox_set_address(self, address=0):
        """
        This API defines the offset location within the DLPC350 mailboxes to write data into or to read data from.
        :param address: Defines the offset within the selected (opened) LUT to write/read data to/from (0-127).
        :type address: int
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        address = bits_to_bytes(conv_len(address, 8))
        result = self.command('w', 0x00, 0x1a, 0x32, address)
        return result
        
    def read_mailbox_address(self):
        """
        Function to read current mailbox address
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.command('r', 0x00, 0x1a, 0x32, [])
        return result
        
    def send_img_lut(self, index_list, address):
        """
        The following parameters: display mode, trigger mode, exposure, 
        and frame rate must be set up before sending any mailbox data.
        If the mailbox is opened to define the flash image indexes, list the index numbers in the mailbox. 
        For example, if image indexes 0 through 3 are desired, write 0x0 0x1 0x2 0x3 to the mailbox.
        :param index_list: image index list to be written
        :param address: starting offset location within the DLPC350 mailboxes to write data
        :type index_list: list
        :type address: int
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        result = self.open_mailbox(1)
        result &= self.mailbox_set_address(address=address)
        result &= self.command('w', 0x00, 0x1a, 0x34, index_list)
        result &= self.open_mailbox(0)
        return result
        
    def pattern_lut_payload_list(self,
                                 trig_type,
                                 bit_depth,
                                 led_select, 
                                 swap_location_list, 
                                 image_index_list,
                                 pattern_num_list,
                                 do_invert_pat=False,
                                 do_insert_black=True,
                                 do_trig_out_prev=False):
        """
        Helper function to create payload for creating pattern LUT written using send_pattern_lut.
        :param trig_type:the trigger type for the pattern
                        :0: Internal trigger.
                        :1: External positive.
                        :2: External negative.
                        :3: No Input Trigger (Continue from previous; Pattern still has full exposure time).
        :param bit_depth: desired bit-depth
        :param led_select: LEDs that are on
        :param swap_location_list: list of indices to perform a buffer swap if true.
        :param image_index_list: projector pattern sequence to create and project.
        :param pattern_num_list:  pattern number for each pattern in image_index_list.
        :param do_invert_pat: invert pattern if true.
        :param do_insert_black: Insert black-fill pattern after current pattern if true.
                                This setting requires 230 μs of time before the start of the next pattern.
        :param do_trig_out_prev:Trigger Out 1 will continue to be high when set true.
        :type trig_type: int
        :type bit_depth: int
        :type led_select: int
        :type swap_location_list:list
        :type image_index_list: list
        :type pattern_num_list: list
        :type do_invert_pat: bool
        :type do_insert_black: bool
        :type do_trig_out_prev: bool
        :return payload_list: hex payload for each pattern entry
        :rtype: list
        """
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
            byte_0 = conv_len(pattern_num_list[i], 6) + trig_type
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
        Mailbox content to set up pattern definition. See table 2-65 in programmer's guide for detailed description of
        pattern LUT entries; Table 2-69 for Pattern Definition.
        :param trig_type: Select the trigger type for the pattern.
                              :0: Internal trigger.
                              :1: External positive.
                              :2: External negative.
                              :3: No Input Trigger (Continue from previous; Pattern still has full exposure time).
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
        :param led_select: Choose the LEDs that are on (bit flags b0 = Red, b1 = Green, b2 = Blue)
                          :0: 0b000 No LED (Pass through)
                          :1: 0b001 Red
                          :2: 0b010 Green
                          :3: 0b011 Yellow (Green + Red)
                          :4: 0b100 Blue
                          :5: 0b101 Magenta (Blue + Red)
                          :6: 0b110 Cyan (Blue + Green)
        :param swap_location_list: list of indices to perform a buffer swap if true.
        :param image_index_list: projector pattern sequence to create and project.
        :param pattern_num_list: Pattern number (0 based index). For pattern number ``0x3F``, there is no pattern display.
                                 The maximum number supported is 24 for 1 bit-depth patterns. Setting the pattern
                                 number to be 25, with a bit-depth of 1 will insert a white-fill pattern.
                                 These patterns will have the same exposure time as defined in the Pattern Display
                                 Exposure and Frame Period command. Table 2-70 in the programmer's guide
                                 illustrates which bit planes are illuminated by each pattern number.
        :param starting_address: Defines the offset within the selected (opened) LUT to write/read data to/from (0-127).
        :param bool do_invert_pat: Invert pattern if true.
        :param bool do_insert_black: Insert black-fill pattern after current pattern if true.
                                     This setting requires 230 us of time before the start of the next pattern.
        :param do_trig_out_prev: Trigger Out 1 will continue to be high when set true. There will be no falling edge
                                 between the end of the previous pattern and the start of the current pattern.
                                 Exposure time is shared between all patterns defined under a common trigger out.
                                 This setting cannot be combined with the black-fill pattern.
        :type trig_type: int
        :type bit_depth: int
        :type led_select: int
        :type swap_location_list:list
        :type image_index_list: list
        :type pattern_num_list: list
        :type starting_address: int
        :type do_invert_pat: bool
        :type do_insert_black: bool
        :type do_trig_out_prev: bool
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        
        payload_flat_list = self.pattern_lut_payload_list(trig_type,
                                                          bit_depth,
                                                          led_select, 
                                                          swap_location_list, 
                                                          image_index_list, 
                                                          pattern_num_list, 
                                                          do_invert_pat,
                                                          do_insert_black,
                                                          do_trig_out_prev)
        if payload_flat_list:
            result = self.open_mailbox(2) 
            result &= self.mailbox_set_address(address=starting_address)
            result &= self.command('w', 0x00, 0x1a, 0x34, payload_flat_list)
            result &= self.open_mailbox(0) 
            result &= self.read_mailbox_info()  # to update the image and pattern LUT table
        else:
            result = False
        return result
        
    def pattern_display(self, action='start'):
        """
        This API starts or stops the programmed patterns sequence.
        :param action: Pattern Display Start/Stop Pattern Sequence.
        :type action: str
        :return result:True if successful, False otherwise.
        :rtype result: bool
        """
        actions = ['stop', 'pause', 'start']
        if action in actions:
            action_no = actions.index(action)
            result = self.command('w', 0x00, 0x1a, 0x24, [action_no])
        else:
            result = False
            print("\n Invalid action keyword, must be one of {'stop', 'pause', 'start'}")
        return result
         
    def read_mailbox_info(self):
        """
        This function reads image LUT table and pattern LUT contents
        :return result: True if all steps executed correctly
        :rtype result: bool
        """
        # Read image table
        result = self.open_mailbox(1)
        result &= self.mailbox_set_address(address=0)
        result &= self.command('r', 0x00, 0x1a, 0x34, [])
        image_ans = self.ans[4:].tolist()
        result &= self.open_mailbox(0)
        
        # Read pattern LUT
        result &= self.open_mailbox(2)
        result &= self.mailbox_set_address(address=0)
        result &= self.command('r', 0x00, 0x1a, 0x34, [])
        pattern_ans = self.ans[4:].tolist()
        result &= self.open_mailbox(0)
        if result:
            self.image_LUT_entries = image_ans
            self.pattern_LUT_entries = pattern_ans
        else:
            self.image_LUT_entries = None
            self.pattern_LUT_entries = None
        return result
        
def get_image_LUT_swap_location(image_index_list):
    """
    Function creates buffer swap location index list and image index according to image_index_list pattern list.
    DLPC350 stores two 24-bit frames in its internal memory buffer.This 48 bit-plane
    display buffer allows the DLPC350 to send one 24-bit buffer to the DMD array while the second
    buffer is filled from flash or streamed in through the 24-bit parallel RGB or FPD-link interface.
    In streaming mode, the DMD array displays the previous 24-bit frame while the current frame fills the second 24-bit
    frame of the display buffer. Once a 24-bit frame is displayed, the buffer rotates providing the next 24-bit
    frame to the DMD.
    If there are only 2 images(24bits each), DLPC350 will fill buffer0, swap, then fill buffer1, but no projecting.
    To get the correct order of pattern, need to do switch the order of pattern(set temp1 as entry0 and set temp0 as entry1).
    If more than 2 images in LUT, DLPC350 only load half of the internal buffer, the projector plot same order as LUT. 
    :param image_index_list:  projector pattern sequence to create and project.
    :type image_index_list: list
    :return image_LUT_entries: user image index LUT list.
    :return swap_location_list: list of indices to perform a buffer swap if true.
    :rtype image_LUT_entries: list
    :rtype swap_location_list: list
    """
    swap_location_list = [0] + [i for i in range(1, len(image_index_list)) if image_index_list[i] != image_index_list[i-1]]
    image_LUT_entries = [image_index_list[i] for i in swap_location_list]

    if len(image_LUT_entries) == 2:    
        temp = image_LUT_entries[0:2].copy()
        image_LUT_entries[0] = temp[1]
        image_LUT_entries[1] = temp[0]
        
    return image_LUT_entries, swap_location_list

def LUT_verification(image_index_list,
                     image_LUT_entries_read,
                     lut_read):
    """
    This function compares the projector LUT with the user requested LUT to verify the correct LUT is generated in the projector.
    The projector has two LUTs, one with only image index order, and the other with the image bit plane order.
    :param image_index_list: user requested LUT
    :param image_LUT_entries_read: projector image index read from projector
    :param lut_read: projector LUT read from projector.
    :type image_index_list: list
    :type image_LUT_entries_read:list
    :type lut_read:list
    :return result:True if successful, False otherwise.
    :rtype result: bool
    """
    
    swap_location_list_read = []
    for i, k in enumerate(lut_read[2::3]):
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

    num_repeats = [swap_location_list_read[i] - swap_location_list_read[i-1] for i in range(1, len(swap_location_list_read))] + [num_of_patterns - swap_location_list_read[-1]]
    image_index_list_recovered = []
    for i, num in enumerate(num_repeats):
        image_index_list_recovered.extend([image_LUT_entries_temp[i]] * num)
        
    if image_index_list == image_index_list_recovered:
        print("\nRecovery successful")
        result = True
    else:
        print("\nRecovery test failed")
        result = False
    
    return result

def current_setting():
    """
    Function to read the current settings of the projector.
    :return result: True if all steps executed correctly
    :rtype result: bool
    """
    with connect_usb() as lcr:
        # to stop current pattern sequence mode
        result = lcr.pattern_display('stop')
        if result:
            lcr.pretty_print_status()
            # do validation and project current LUT patterns
            result &= lcr.start_pattern_lut_validate()
            if result:
                result &= lcr.pattern_display('start')
            else:
                print("ERROR: Projector cannot be started")
                result &= False
        else:
            print("ERROR: Projector cannot be stopped.")
            result &= False
        
    return result

def forge_bmp(single_channel_image_list, savedir, convertRGB=True):
    """
    Function to create list of 3 channel (24 bit) image from list of single channel (8 bit) images.
    :param single_channel_image_list : list of 8 bit images
    :param savedir : directory for saving the file
    :param convertRGB: If set each image will be RGB otherwise BGR
    :type single_channel_image_list: list
    :type savedir : str
    :type convertRGB: bool
    :return three_channel_list: list of three channel image (color image)
    :rtype three_channel_list: list
    """
        
    image_array = np.empty((single_channel_image_list[0].shape[0], single_channel_image_list[0].shape[1], 3))
    three_channel_list = []
    count = 0
    for j, i in enumerate(single_channel_image_list):
        image_array[:, :, (j % 3)] = i
        if j % 3 == 2:  # 0,1,2
            if convertRGB:
                image_array[:, :, [0, 1, 2]] = image_array[:, :, [2, 0, 1]]  # changing channel for saving for projector (RGB)
            cv2.imwrite(os.path.join(savedir, "image_%d.bmp" % count), image_array)
            three_channel_list.append(image_array)
            image_array = np.empty((single_channel_image_list[0].shape[0], single_channel_image_list[0].shape[1], 3))
            count += 1
            
    if len(single_channel_image_list) % 3 != 0:
        print("Warning: Last image in the list is a %d channel image" % (len(single_channel_image_list) % 3))
            
    return three_channel_list 

def forge_fringe_bmp(savedir, 
                     pitch_list,
                     N_list,
                     type_unwrap,
                     phase_st,
                     inte_rang, direc='v',
                     calib_fringes=False,
                     proj_width=912,
                     proj_height=1140):
    """
    Function to creat image deck of fringe patterns used for phase shift fringe projection.
    If  calib_fringes is set to True fringe deck in both vertical and horizontal directions used mainly for calibration
    is created.
    :param savedir: path to save the color pattern images as bmp which can be saved into projector flash
    :param pitch_list: List of number of pixels per fringe period.
    :param N_list: List of number of steps for each pitch.
    :param type_unwrap:Type of temporal unwrapping to be applied.
                      'phase' = phase coded unwrapping method,
                      'multifreq' = multi-frequency unwrapping method
                      'multiwave' = multi-wavelength unwrapping method.
    :param phase_st: Starting phase. To apply multifrequency and multiwavelength temporal unwrapping
                     starting phase should be zero. Whereas for phase coding temporal
                     unwrapping starting phase should be -π.
    :param inte_rang, direc: Visually vertical (v) or horizontal(h) fringe pattern.
    :param calib_fringes: If set with create fringes in both direction.
    :param proj_width: width of projector
    :param proj_height: height of projector
    :type savedir: str
    :type pitch_list: list
    :type N_list: list
    :type type_unwrap: str
    :type phase_st: int
    :type inte_rang: list
    :type direc: str
    :type calib_fringes: bool
    :type proj_width: int
    :type proj_height: int
    :return fringe_bmp_list: list of patterns stacked as 3 channel image.
    :rtype fringe_bmp_list: list
    """
    
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
    
    fringe_bmp_list = forge_bmp(fringe_array, savedir, convertRGB=True)
    
    return fringe_bmp_list

def proj_single_img(image_index,
                    exposure_period=27314,
                    frame_period=27314):
    
    """
    Function projects single image with given flash image index in repeat mode.
    :param image_index: index of image on projector flash.
    :param exposure_period: projector exposure period.
    :param frame_period: projector frame period.
    :type image_index: int
    :type exposure_period: int
    :type frame_period: int
    :return result:True if successful, False otherwise.
    :rtype result: bool
    """
  
    with connect_usb() as lcr:
        try:
            result = True
            result &= lcr.pattern_display('stop')
            result &= lcr.set_pattern_config(num_lut_entries=1,
                                             do_repeat=True,
                                             num_pats_for_trig_out2=1,
                                             num_images=1)
            result &= lcr.set_exposure_frame_period(exposure_period=exposure_period, 
                                                    frame_period=frame_period)            
            # To set image LUT
            result &= lcr.send_img_lut(index_list=[image_index],
                                       address=0)   
            
            # To set pattern LUT table
            result &= lcr.send_pattern_lut(trig_type=0,
                                           bit_depth=8,
                                           led_select=0b111,
                                           swap_location_list=[0],
                                           image_index_list=[image_index],
                                           pattern_num_list=[0],
                                           starting_address=0,
                                           do_invert_pat=False,
                                           do_insert_black=False,
                                           do_trig_out_prev=False)
            lcr.pretty_print_status()
            result &= lcr.start_pattern_lut_validate()
            if result:
                result &= lcr.pattern_display('start')
                input("Press Enter to stop...")
                result &= lcr.pattern_display('stop')                
            else:
                result &= False
        except (Exception,):
            print("An exception occurred")
            result &= False        
    return result

def proj_pattern_LUT(image_index_list, 
                     pattern_num_list, 
                     exposure_period=27084,
                     frame_period=33334,
                     pprint_proj_status=True):
    """
    This function is used to create look up table and project the sequence for the projector based on the 
    image_index_list (sequence) given.
    :param image_index_list : The sequence to be created.
    :param pattern_num_list : The pattern number sequence to be created.
    :param exposure_period: Exposure time in microseconds (4 bytes)
    :param frame_period: Frame period in microseconds (4 bytes).
    :param pprint_proj_status: If set will print projector's current status.
    :return result:True if successful, False otherwise.
    :rtype result: bool
    """
    
    image_LUT_entries, swap_location_list = get_image_LUT_swap_location(image_index_list)    
    proj_timing_start = perf_counter_ns()
    with connect_usb() as lcr:
        try:
            result = True
            result &= lcr.pattern_display('stop')        
            result &= lcr.set_pattern_config(num_lut_entries=len(image_index_list),
                                             do_repeat=False,
                                             num_pats_for_trig_out2=len(image_index_list),
                                             num_images=len(image_LUT_entries))
            result &= lcr.set_exposure_frame_period(exposure_period=exposure_period, 
                                                    frame_period=frame_period)
            # To set image LUT
            result &= lcr.send_img_lut(index_list=image_LUT_entries,
                                       address=0)
            # To set pattern LUT table    
            result &= lcr.send_pattern_lut(trig_type=0,
                                           bit_depth=8,
                                           led_select=0b111,
                                           swap_location_list=swap_location_list,
                                           image_index_list=image_index_list,
                                           pattern_num_list=pattern_num_list,
                                           starting_address=0)
            
            if pprint_proj_status:  # Print all projector current attributes set
                lcr.pretty_print_status()
            image_LUT_entries_read = lcr.image_LUT_entries[0:len(image_LUT_entries)]
            temp = 3 * lcr.num_lut_entries  # each lut entry has 3 bytes
            lut_read = lcr.pattern_LUT_entries[0:temp]
            # start validation
            result &= lcr.start_pattern_lut_validate()
            # Check validation status
            if result:   
                result &= lcr.pattern_display('start')
        except (Exception,):
            print("An exception occurred")
            result = False
    proj_timing_end = perf_counter_ns() 
    proj_timing = (proj_timing_end - proj_timing_start)/1e9    
    print('Projector Timing:%.3f sec' % proj_timing)
    result &= LUT_verification(image_index_list,
                               image_LUT_entries_read, 
                               lut_read)
    return result


def main():
    """
    Main function example.
    """
    option = input("Please choose: 1 -- project single image; 2 -- project pattern sequence\n")
    result = True
    if option == '1':
        result &= proj_single_img(image_index=21)
    if option == '2':
        image_index_list = np.repeat(np.arange(0, 5), 3).tolist()
        pattern_num_list = [0, 1, 2] * len(set(image_index_list))
        result &= proj_pattern_LUT(image_index_list,
                                   pattern_num_list)
    return result


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

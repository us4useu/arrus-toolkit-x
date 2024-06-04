import math

'''
Author: Damian Cacko, damian.cacko@us4us.eu
Date: 11.03.2023

Notes:
This is a simple tool to find possible frequency outputs for us4R and us4R-lite systems.
Due to limited timing precision of digital transmit beamformer, the possible output frequecies is a limited set of values.
The lower is output frequency, the finer is setting.
Please set the type of OEM module of your system in the variable below and use functions provided below. Examples are shown in the end of this file
'''
OEM_TYPE = 'OEM+'   # set 'OEM' or 'OEM+'


# Funcions
def getTxFreq(TxFreq):
    '''
    Input parameters:
    TxFreq: desired output frequency in [MHz]
    
    Returns: (ClosestTxFreq, ClosestTxFreq_Alternative)
    f1: possible output frequency [MHz], closest to desired
    f2: second closest possible output frequency [MHz].
    '''
    if(OEM_TYPE == 'OEM'):
        txClk = 130.0
    elif(OEM_TYPE == 'OEM+'):
        txClk = 195.0
     
    req_period_ns = 1000.0 / TxFreq
    clk_period_ns = 1000.0 / txClk
    
    hcCks0 = math.ceil(req_period_ns / clk_period_ns / 2)
    hcCks1 = math.floor(req_period_ns / clk_period_ns / 2)
    
    if(OEM_TYPE == 'OEM'):
        if(hcCks0 < 1):
            hcCks0 = 1
        elif(hcCks0 > 64):
            print('Requested frequency is below possible range. Returning minimum possible.')
            hcCks0 = 64
       
        if(hcCks1 < 1):
            hcCks1 = 1
        elif(hcCks1 > 64):
            hcCks1 = 64    
        
    elif(OEM_TYPE == 'OEM+'):
        if(hcCks0 < 2):
            hcCks0 = 2

        if(hcCks1 < 2):
            hcCks1 = 2         
    
    f1 = 1000.0 / (hcCks0 * 2 * clk_period_ns )
    f2 = 1000.0 / (hcCks1 * 2 * clk_period_ns )
    
    if(abs(f1-TxFreq) < abs(f2-TxFreq)):
        return (f1, f2) 
    else:
        return (f2, f1)

    
def printTxFrequenciesList():
    '''
    Input parameters:
    None.
    
    Returns: 
    - a list of possible output frequencies. 
    For OEM, all possible frequencies are printed.
    For OEM+, only frequencies above 1 MHz are returned, since below 1 MHz setting is fine. Output frequency can be almost any low.
    '''
    
    TxFreq = []
    if(OEM_TYPE == 'OEM'):
        txClk = 130.0
        clk_period_ns = 1000.0 / txClk
        for i in range(1, 65, 1):
            TxFreq.append(1000.0 / (i * 2 * clk_period_ns ))
        
    elif(OEM_TYPE == 'OEM+'):
        txClk = 195.0
        clk_period_ns = 1000.0 / txClk
        for i in range(2, 98, 1):
            TxFreq.append(1000.0 / (i * 2 * clk_period_ns ))
        
    N = len(TxFreq)
    for j in range(N-1, -1, -1):
        print(TxFreq[j])
        
    
# Examples
(f1, f2) = getTxFreq(1.2)
print(f1)
print(f2)
print('---------------')

printTxFrequenciesList()

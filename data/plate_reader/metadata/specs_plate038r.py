plate_name = 'plate038r'

growth_file = "plate038i_600.txt"
miller_csv = plate_name + "_420.txt"
optics_csv = plate_name + "_550.txt"

raw_growth = "2017.5.26_lacstar_cminusspacing_plate2_induced_335_OD600.txt"
raw_miller = "2017.5.30_lacstar_cminusspacing_plate_induced_repeat_miller_420.txt"
raw_optics = "2017.5.30_lacstar_cminusspacing_plate_induced_repeat_miller_550.txt"

num_rows = 8
num_cols = 12
date = '17.05.30'

stop_time = 60
start_time = 6

cultures_dict = {
	0:'RDM',
	1:'b1A5',
	2:'b1H6',
	3:'b1E4',
	4:'b3D3',
	5:'b3D4',
	6:'b3D6',
	7:'b3D8',
	8:'b3E1',
	9:'b3E3',
	10:'b3E5',
	11:'b3E7',
	12:'b3E9',
	13:'b3F2',
	14:'b3F4',
	15:'b3F6',
	16:'b3F8',
	17:'b3G1',
	18:'b3G3',
	19:'b3G5',
	20:'b3G7',
	21:'b3G9',
	22:'b3H2',
	23:'b3H4',
	24:'b1F1'
	}
	
num_strains = max(cultures_dict.keys())
	
culture_locations_string = '''
1	0	5	0	9	0	13	0	17	0	21	0
1	3	5	7	9	11	13	15	17	19	21	23
1	3	5	7	9	11	13	15	17	19	21	23
0	3	0	7	0	11	0	15	0	19	0	23
2	0	6	0	10	0	14	0	18	0	22	0
2	4	6	8	10	12	14	16	18	20	22	24
2	4	6	8	10	12	14	16	18	20	22	24
0	4	0	8	0	12	0	16	0	20	0	24
'''

camp_concentrations_string = '''
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
250	250	250	250	250	250	250	250	250	250	250	250
'''

culture_volumes_string = '''
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
50	50	50	50	50	50	50	50	50	50	50	50
'''


import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import keyboard

def readFile(filename, x=2000):
	#wfdb automatically reads 'filename.hea' and 'filename.dat'
	signals, fields = wfdb.rdsamp(filename)

	print(f"""\n--- Header Metadata ---
		Sampling Frequency: {fields['fs']} Hz
		Signal Shapes: {signals.shape}
		Channel Names: {fields['sig_name']}
		Units: {fields['units']}\n
	""")

	#Convert NumPy matrix into DataFrame
	Data_File = pd.DataFrame(signals, columns=fields['sig_name'])

	#Add timestamp column
	Data_File['Time_Sec'] = Data_File.index / fields['fs']

	#Print first few rows
	print(str(Data_File.head()) + "\nClose waveform window to continue")

	#Plots x datapoints of the waveforms
	wfdb.plot_items(
		signal=signals[:x, :],
		title='ECG Waveform Analysis',
		fs=fields['fs'],
		time_units='seconds'
	)

Filename_Structures = ["ARR_", "NR_"]
Database_Directories = ["Fetal_ECG_Database_"]
fStructure = Filename_Structures[0]

File_Quantity = 12
Selected_Directory = 1

directory = str(Database_Directories[0]) + str(Selected_Directory) + "/"

for filenumber in range(1, File_Quantity + 1):
	readFile(filename=(directory + fStructure + str(filenumber).zfill(2)))
	#if filenumber != File_Quantity:
	#	input("Press [Enter] to continue")

print("##File Range Completed##")
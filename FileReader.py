import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import keyboard

class WFDB:

	def readFile(self, filename, x=5000):
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

class CSV():

	def readFile():
		pass


def selectListItem(list):
	ask = "Select Directory Index\n"
	for item in list:
		it2 = ""
		itemRange = 2
		if isinstance(item[2], WFDB):
			it2 = "WFDB"
		elif isinstance(item[2], CSV):
			it2 = "CSV"
		else:
			itemRange = len(item)
		ask += "-\t" + str(item[:itemRange]) + f" {it2}\n"

	ask += "["
	for index in range(0, len(list)):
		ask += str(index)
		if index != len(list) - 1:
			ask += ","
	ask += "] > "

	index = int(input(ask))
	return list[index]


#List of lists containing dataset naming conventions, the number of records, and the file type
#Note: the second list only contains one record labelled 'p10143'

File_Structures = [
	[["ARR_", "NR_"], 12, WFDB()],
	["p", 10143, WFDB()],
	["r", 5, CSV()],
	["high_precision_fecg_signal_amplification_dataset", 1, CSV()]
]


Database_Directories = ["Fetal_ECG_Database_1"]


Selected_Dataset = selectListItem(File_Structures)
File_Quantity = Selected_Dataset[1]
directory = selectListItem(Database_Directories) + "/"

if File_Quantity != 10143:
	for convention in Selected_Dataset[0]:
		for filenumber in range(1, File_Quantity + 1):
			Selected_Dataset[2].readFile(filename=(directory + convention + str(filenumber).zfill(2)))
			#if filenumber != File_Quantity:
			#	input("Press [Enter] to continue")

	print("##File Range Completed##")

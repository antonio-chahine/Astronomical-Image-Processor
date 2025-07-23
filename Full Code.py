from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, gaussian_filter
from scipy.optimize import curve_fit
from astropy.stats import sigma_clipped_stats
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

params = {
   'axes.labelsize': 18,
   'font.size': 18,
   'font.family': 'sans-serif', 
   'font.serif': 'Arial', 
   'legend.fontsize': 12,
   'xtick.labelsize': 16,
   'ytick.labelsize': 16, 
   'figure.figsize': [8.8, 8.8/1.618] 
} 
plt.rcParams.update(params)

hdulist = fits.open("mosaic.fits")
data = hdulist[0].data
#data = gaussian_filter(data, sigma=1)


#BACKGROUND MEAN AND S.D CALCULATION
lower = 3355
upper = 3480

hist_data = data.flatten()
filtered_data = hist_data[(hist_data >= lower) & (hist_data<=upper)]

def gaus(x,m,sig,A):
    return A*np.exp(-0.5*((x-m)/sig)**2)

counts, bin_edges, _ = plt.hist(filtered_data, bins=round((upper-lower)/2),alpha=0.7,color='#008080', label='Counts', edgecolor='#2F4F4F')
bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
po_hist,po_cov_hist=curve_fit(gaus,bin_centers,counts, p0=[3420,30,40000])
x = np.linspace(lower, upper,1000)
plt.plot(x,gaus(x,po_hist[0],po_hist[1],po_hist[2]),lw=5,color='#FF8C00',label='Fitted Gaussian')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend()
plt.title('Background Counts')
plt.savefig("bg.pdf", bbox_inches='tight')
plt.show()



print('passed histogram')

#MASK + REMOVE EDGES + STANDARD DEVIATION MASK

#%%

#total 10 regions. sift out grey boxes of 3421 counts (region 2 and region 9 need extra manipulation. See comments)

region0_xleft = 0
region0_xright = 115
region0_ybottom = 4519
region0_ytop = 4611

region1_xleft = 0
region1_xright = 2570
region1_ybottom = 4610
region1_ytop = 4611

#region 2 is long thin bar ranging from 2-6px thickness, but we just use code to find pixels with flux = 3421
region2_xleft = 0
region2_xright = 6 + 1
region2_ybottom = 0
region2_ytop = 4611

region3_xleft = 7
region3_xright = 26 + 1
region3_ybottom = 0
region3_ytop = 423

region4_xleft = 27
region4_xright = 96 + 1
region4_ybottom = 0
region4_ytop = 403

region5_xleft = 97
region5_xright = 117 + 1
region5_ybottom = 0
region5_ytop = 119

region6_xleft = 118
region6_xright = 407 + 1
region6_ybottom = 0
region6_ytop = 100

region7_xleft = 408
region7_xright = 424
region7_ybottom = 0
region7_ytop = 21 + 1

region8_xleft = 0
region8_xright = 2570
region8_ybottom = 0
region8_ytop = 1

region9_xleft = 2471 + 1
region9_xright = 2570
region9_ybottom = 0
region9_ytop = 116 + 1

#region 10 is long thin bar ranging from 1-2px thickness, but we just use code to find pixels with flux = 3421
region10_xleft = 2569
region10_xright = 2570
region10_ybottom = 0
region10_ytop = 4611

region11_xleft = 2166 + 1
region11_xright = 2570
region11_ybottom = 4514 + 1
region11_ytop = 4611

region12_xleft = 2478 + 1
region12_xright = 2570
region12_ybottom = 4212 + 1
region12_ytop = 4611

xlefts = np.array([region0_xleft, region1_xleft, region2_xleft, region3_xleft, region4_xleft, region5_xleft, region6_xleft, region7_xleft, region8_xleft, region9_xleft, region10_xleft, region11_xleft, region12_xleft])
xrights = np.array([region0_xright, region1_xright, region2_xright, region3_xright, region4_xright, region5_xright, region6_xright, region7_xright, region8_xright, region9_xright, region10_xright, region11_xright, region12_xright])
ybottoms = np.array([region0_ybottom, region1_ybottom, region2_ybottom, region3_ybottom, region4_ybottom, region5_ybottom, region6_ybottom, region7_ybottom, region8_ybottom, region9_ybottom, region10_ybottom, region11_ybottom, region12_ybottom])
ytops = np.array([region0_ytop, region1_ytop, region2_ytop, region3_ytop, region4_ytop, region5_ytop, region6_ytop, region7_ytop, region8_ytop, region9_ytop, region10_ytop, region11_ytop, region12_ytop])




#VARIABLE THRESHOLD
box_width = 100
box_height = 100

image_height, image_width = data.shape
# Step 3: Initialize an empty list to store background noise statistics

background_stats = []
threshold_value_mask = np.zeros(data.shape)
threshold_value_sd = np.zeros(data.shape)
threshold_value_mean = np.zeros(data.shape)

# Step 4: Iterate over the image, extracting segments
for y in range(0, image_height, box_height):
    for x in range(0, image_width, box_width):
        # Step 4a: Calculate the actual width and height of the segment
        # This ensures that segments at the edges are smaller if they exceed the image boundaries
        current_segment_height = min(box_height, image_height - y)
        current_segment_width = min(box_width, image_width - x)

        # Step 4b: Extract the segment
        segment = data[y:y + current_segment_height, x:x + current_segment_width]

        # Step 5: Calculate the background noise using sigma-clipped statistics
        mean, median, stddev = sigma_clipped_stats(segment, sigma=3)

        # Step 6: Store the results for this segment
        background_stats.append({
            'x': x,
            'y': y,
            'segment_width': current_segment_width,
            'segment_height': current_segment_height,
            'mean': mean,
            'median': median,
            'stddev': stddev
        })

        
    for segment in background_stats:
        x = segment['x']
        y = segment['y']
        segment_width = segment['segment_width']
        segment_height = segment['segment_height']
        mean = segment['mean']
        stddev = segment['stddev']
        
        threshold_value_mask[y:y+segment_height, x:x+segment_width] = mean + 3 * stddev
        threshold_value_sd[y:y+segment_height, x:x+segment_width] = stddev
        threshold_value_mean[y:y+segment_height, x:x+segment_width] = mean


#%%

#threshold mask 
threshold_mask = np.zeros(data.shape)
for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] > threshold_value_mask[i][j]: #threshold
            threshold_mask[i][j] = 1
      
#defective corner mask
defect_mask = np.ones(data.shape)
for region_number in range(13):
    x_left = xlefts[region_number]
    x_right = xrights[region_number]
    y_bottom = ybottoms[region_number]
    y_top = ytops[region_number]
    for i in range(y_bottom, y_top):
        for j in range(x_left, x_right):
            if data[i][j] <= 3421:
                defect_mask[i][j] = 0    


mask = threshold_mask * defect_mask


#2570 x 4611
a, b, c, d = 0, 4611, 0, 2570
data = data[a:b, c:d]
mask = mask[a:b, c:d]

'''
distance = ndi.distance_transform_edt(mask)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((4, 4)), labels=mask.astype(int))
markers = ndi.label(local_maxi)[0]
labeled_mask = watershed(-distance, markers, mask=mask)
num_objects = len(np.unique(labeled_mask)) - 1
'''
print('passed threshold')


#%%


#CREATE OBJECT CATALOG

labeled_mask, num_objects = label(mask)
plt.imshow(labeled_mask, cmap='nipy_spectral', interpolation='none', origin='lower')
plt.show()

object_catalog = []
for obj_label in range(1, num_objects + 1):
    coordinates = np.argwhere(labeled_mask == obj_label)

    object_catalog.append({
        'label': obj_label,
        'coordinates': coordinates,
        'size': len(coordinates),
        'brightness': 0,
        'brightness_uncertainty': 0
        })


sizes = [item['size'] for item in object_catalog]
_=plt.hist(sizes, bins=100,range=(1,10))
plt.show()

print('passed object catalog')
#%%

#FILTERING

total = 0
total_squared = 0
for obj in object_catalog:
    total += len(obj['coordinates'])
    total_squared += (len(obj['coordinates']))**2
mean = total/num_objects
standard_deviation = np.sqrt((total_squared/num_objects)-mean**2)
sizes = [len(obj['coordinates']) for obj in object_catalog]
median_size = np.median(sizes)


filtered_object_catalog = []

total = 0

for obj in object_catalog:
    coordinates = obj['coordinates']
  
    remove_object = False
    
    for coord in coordinates:
        x_value, y_value = coord
        pixel_value = data[x_value][y_value]
        
        
        if pixel_value > 50000:
            remove_object = True
            break
        
        #object may have threshold value on outer part doesnt mean we get rid of the whole object!
        #need to be if the sum of pixel values is below the threshold
    
    total_brightness = 0
    for coord in coordinates:
        x_value, y_value = coord
        pixel_value = data[x_value][y_value]
        total_brightness += pixel_value - threshold_value_mask[x_value][y_value]
        
    if total_brightness <= 0:
        remove_object = True
           
        
    if len(coordinates) <= 5:
        remove_object = True
        
    
    if len(coordinates) > (mean + 2.5*standard_deviation):
        remove_object = True

    
    # If object passes checks
    if not remove_object:
        filtered_object_catalog.append(obj)

# Replace the original catalog with the filtered one
object_catalog = filtered_object_catalog


print('passed filtering')


#CREATE THE FILTERED MASK PLOT
filtered_mask = np.zeros_like(mask)

for obj in object_catalog:
    coordinates = obj['coordinates']
    for coord in coordinates:
        x_value, y_value = coord
        filtered_mask[x_value, y_value] = 1  

plt.imshow(filtered_mask, cmap='binary', interpolation='none', origin='lower')
plt.title('Filtered Image')
plt.show()





#FINDS THE MAGNITUDE OF EACH OBJECT, UNCERTAINTY, ADDS TO CATALOG

for i in range(0,len(object_catalog)):
    total_brightness = 0
    background_uncertainty = 0
    total_brightness_uncertainty = 0
    coordinates = object_catalog[i]['coordinates']
    for j in range(0,len(object_catalog[i]['coordinates'])):
        x_value, y_value = coordinates[j]
        pixel_value = data[x_value][y_value]
        total_brightness += pixel_value - threshold_value_mask[x_value][y_value]
        
        
        background_uncertainty += (threshold_value_sd[x_value][y_value])**2
        total_brightness_uncertainty += np.sqrt(pixel_value)
        
    mag = 25.3 - (2.5*np.log10((total_brightness)))
    object_catalog[i]['brightness'] = mag
    
    
    background_uncertainty = np.sqrt(background_uncertainty)
    total_brightness_uncertainty = np.sqrt(total_brightness_uncertainty)
    mag_uncertainty = 25.3 - (2.5*np.log10(total_brightness_uncertainty))
    object_catalog[i]['brightness_uncertainty'] = np.sqrt((mag_uncertainty/mag)**2+(background_uncertainty/threshold_value_mask[x_value][y_value])**2)


#%%




print('number of object:', len(object_catalog))

##ACTUAL COUNTS GRAPH

brightness_array = [item['brightness'] for item in object_catalog]
_=plt.hist(brightness_array, bins=15, range=(12,25),color='#008080', label='Counts', edgecolor='#2F4F4F')
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
plt.legend()
plt.title('Total Counts')
plt.savefig("bg.pdf", bbox_inches='tight')
plt.show()
plt.show()


# Create brightness array from the catalog
brightness_array = [item['brightness'] for item in object_catalog]

# Calculate histogram data (counts and bin edges)
counts, bin_edges = np.histogram(brightness_array, bins=13, range=(12, 25))

# Calculate bin centers for plotting error bars
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Calculate the errors as the square root of the counts
errors = np.sqrt(counts)

# Plot the histogram without the error bars
plt.hist(brightness_array, bins=13, range=(12,25), color='#008080', label='Counts', edgecolor='#2F4F4F')

# Add error bars
plt.errorbar(bin_centers, counts, yerr=errors, fmt='none', ecolor='black', capsize=5, label='error', elinewidth=3)

# Set scientific notation for y-axis
plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

# Add legend and title
plt.legend()
plt.title('Total Counts')

# Save the figure
plt.savefig("bg.pdf", bbox_inches='tight')

# Show the plot
plt.show()

##SIZE GRAPH
sizes = [item['size'] for item in object_catalog]
_=plt.hist(sizes, bins=70,range=(0,70))
plt.show()



#CREATING THE GRAPH


#sky area

pixel_scale_x = np.sqrt((2.04461e-07)**2 + (-7.27302e-05)**2)
pixel_scale_y = np.sqrt((-7.27927e-05)**2 + (-1.43607e-07)**2)

image_area = (pixel_scale_x * (d-c)) * (pixel_scale_y * (b-a))


min_brightness = min(item['brightness'] for item in object_catalog)
max_brightness = max(item['brightness'] for item in object_catalog)

x = np.linspace(min_brightness-0.5,max_brightness,round((max_brightness-min_brightness)/0.5))
y = []
y_error = []

x = x[1:20]

for i in range(0,len(x)):
    num = sum(1 for obj in object_catalog if obj['brightness'] <= x[i])
    num_normalised = num/(image_area*0.5)
    y.append(num_normalised)
    
    
    cumulative_uncertainty = np.sqrt(sum(obj['brightness_uncertainty']**2\
        for obj in object_catalog if obj['brightness'] < x[i]))
    cumulative_uncertainty_normalised= cumulative_uncertainty/(image_area*0.5)
    y_error.append(cumulative_uncertainty_normalised)
  
def log_model(x,c):
    return 10**(0.6*x + c)

def log(x,c,m):
    return 10**(m*x + c)

x_fit = x[0:8]
y_fit = y[0:8]
y_error_fit = y_error[0:8]

po_model,po_cov_model=curve_fit(log_model,x_fit,y_fit, p0=[-4], sigma=y_error_fit)
po,po_cov=curve_fit(log,x_fit,y_fit, p0=[-4,0.6], sigma=y_error_fit)


y_error_log = y_error / (np.array(y)*np.log(10))

plt.plot(x,np.log10(y),".", markersize = 8, color='black')
plt.errorbar(x,np.log10(y),y_error_log, fmt='none', capsize=4, ecolor='black')
plt.plot(x,np.log10(log_model(x,po_model[0])),"-", linestyle="dashed", label="Theoretical Model")
plt.plot(x,np.log10(log(x,po[0],po[1])),"-", label="Best Fit")
plt.xlabel('Apparent Magnitude (m)')
plt.ylabel("log N / 0.5mag / deg$^{2}$")
#plt.ylim(2.7,6)
#plt.xlim(11.5,19)
plt.legend()
plt.grid()
plt.title("Cumulative Count Plot")
plt.savefig("final.pdf", bbox_inches='tight')
plt.show()


def chisqr(obs, exp, error):
    chisqr = 0
    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
    return chisqr


y_line = np.log10(log_model(x,po[0]))
                  
print(chisqr(np.log(y), y_line, y_error))




#homogeneous assumption (y-axis scale) larger area
#compare to aperture method
#exposure time!!
#poisson distribution flux
#one phton cut off ccd
#2570 x 4611
#say that out no-evolution model is accruate for the range
#size (histogram) do all objects look like they're in the galaxy
#from simulation: code 95% accurate the model is wrong
#functional form of simulation, does the 5% spread evenly or just at the lower brightnessses
#are galaxies gaussian shape
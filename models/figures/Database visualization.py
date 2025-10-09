#pip install geopandas
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

####################################################MAP of USA#############################################################
# Read the extracted map data for the continental US
usa = gpd.read_file('ne_10m_admin_0_countries_usa.shp')
usa_continental = usa[usa['ADM0_A3'] == 'USA']  # Select only the continental US

# Define regions with their corresponding colors and sample sizes
regions = {
    'D1': {'lon_range': (-129.9241, -100.2167), 'lat_range': (24, 60), 'color': (210/255, 180/255, 140/255, 0.6) , 'samples': 5781},
    'D2': {'lon_range': (-99.9995, -90.3694), 'lat_range': (24, 60), 'color': (125/255, 222/255, 112/255, 0.5) , 'samples': 831},
    'D3': {'lon_range': (-89.9656, -62.2240), 'lat_range': (24, 60), 'color': (34/255, 139/255, 34/255, 0.5), 'samples': 4857},
}

# Create an empty GeoDataFrame to store selected regions
selected_regions = gpd.GeoDataFrame(geometry=[])
for key, value in regions.items():
    lon_range = value['lon_range']
    lat_range = value['lat_range']
    minx, maxx = lon_range
    miny, maxy = lat_range
    rectangle = box(minx, miny, maxx, maxy)
    rectangle_gdf = gpd.GeoDataFrame({'geometry': [rectangle]}, crs=usa.crs)
    clipped = gpd.clip(usa_continental, rectangle_gdf.geometry.values[0])
    if not clipped.empty:
        clipped['region'] = key
        selected_regions = pd.concat([selected_regions, clipped], ignore_index=True)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_extent([-125, -66.5, 24, 50], crs=ccrs.PlateCarree())

# Add background and geographical features
ax.stock_img()
ax.add_feature(cfeature.LAND, facecolor='lightgrey')
ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='white')
ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8, edgecolor='white')
ax.add_feature(cfeature.RIVERS, linewidth=0.5, edgecolor='blue', alpha=0.8)
ax.add_feature(cfeature.LAKES, edgecolor='blue', facecolor='none', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax.add_feature(cfeature.OCEAN, edgecolor='blue', linewidth=0.5)

# Plot different colors by region and add legend content
for key, value in regions.items():
    region_data = selected_regions[selected_regions['region'] == key]
    region_data.plot(ax=ax, color=value['color'], edgecolor='black', label=key)

# Draw the boundary of the continental US
usa_continental.boundary.plot(ax=ax, color='black', linewidth=0.5)

# Add gridlines
grid = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
grid.xlabels_top = False
grid.ylabels_right = False
grid.xlabel_style = {'fontsize': 32, 'fontname': 'Times New Roman'}
grid.ylabel_style = {'fontsize': 32, 'fontname': 'Times New Roman'}

# Add compass
compass_x = -120  # x-coordinate of the compass
compass_y = 27  # y-coordinate of the compass
ax.annotate('', xy=(compass_x, compass_y + 1.5), xytext=(compass_x, compass_y),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(compass_x, compass_y + 1.9, 'N', fontsize=24, ha='center', va='center')

ax.annotate('', xy=(compass_x, compass_y - 1.5), xytext=(compass_x, compass_y),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(compass_x, compass_y - 2.1, 'S', fontsize=24, ha='center', va='center')

ax.annotate('', xy=(compass_x - 2, compass_y), xytext=(compass_x, compass_y),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(compass_x - 2.5, compass_y, 'W', fontsize=24, ha='center', va='center')

ax.annotate('', xy=(compass_x + 2, compass_y), xytext=(compass_x, compass_y),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(compass_x + 2.1, compass_y, 'E', fontsize=24, ha='center', va='center')

# Add legend
handles = [plt.Line2D([0], [0], marker='o', color='w', label=f"{key} Region",
                       markerfacecolor=value['color'], markersize=10) for key, value in regions.items()]
ax.legend(handles=handles, title='Regions', loc='lower right', fontsize=20, title_fontsize=22, frameon=True, handlelength=0.2)

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

plt.show()



################################################# Sample Counts for Each Region ###################################################

# Region names and sample counts
regions = ['D1', 'D2', 'D3']
sample_counts = [5781, 831, 4857]

# Custom colors
colors = [
    (216/255, 197/255, 159/255),  # Adjusted desert shrub
    (179/255, 228/255, 151/255),  # Adjusted grassland
    (108/255, 171/255, 107/255)   # Adjusted temperate deciduous forest
]

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create bar chart
plt.figure(figsize=(12, 6), facecolor='none')
bar_width = 0.6  # Set the width of the bars
bars = plt.bar(regions, sample_counts, color=[c[:3] for c in colors], alpha=1, width=bar_width)  # Set to opaque

# Add title and labels
# plt.title('Sample Counts for Each Region', fontsize=28, pad=40)  # Increase title font size and move it up
plt.xlabel('Regions', fontsize=32)  # Increase x-axis label font size
plt.ylabel('Sample Count', fontsize=32)  # Increase y-axis label font size
plt.yticks(fontsize=28)  # Increase y-axis tick font size
plt.xticks(fontsize=28)  # Increase x-axis tick font size

# Add value labels
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
             f'{bar.get_height()}', ha='center', fontsize=28)  # Increase value label font size
plt.gca().patch.set_facecolor('none')  # Set background to transparent

# Remove borders
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Display the bar chart
plt.show()




######################################### Stacked Number of Segments by Bird Species ######################################

# Data preparation
data = {
    'Bird Species': [f'#{i}' for i in range(10)],
    'D1': [1295, 778, 345, 645, 392, 730, 199, 223, 138, 1038],
    'D2': [54, 166, 12, 123, 50, 9, 107, 94, 29, 187],
    'D3': [839, 1299, 132, 435, 96, 297, 579, 283, 106, 791]
}

df = pd.DataFrame(data)

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Create stacked bar chart
plt.figure(figsize=(17, 8), facecolor='none')

# Draw stacked bar chart
plt.barh(df['Bird Species'], df['D1'], color=colors[0], label='D1')
plt.barh(df['Bird Species'], df['D2'], left=df['D1'], color=colors[1], label='D2')
plt.barh(df['Bird Species'], df['D3'], left=df['D1'] + df['D2'], color=colors[2], label='D3')

# Value labels
for i in range(len(df)):
    plt.text(df['D1'][i] / 2 - 10, i, df['D1'][i], ha='center', va='center', fontsize=26)
    plt.text(df['D1'][i] + df['D2'][i] / 2, i, df['D2'][i], ha='center', va='center', fontsize=26, color='gray')
    plt.text(df['D1'][i] + df['D2'][i] + df['D3'][i] - 10, i, df['D3'][i], ha='right', va='center', fontsize=26)
    total = df.iloc[i, 1:].sum()
    plt.text(total + 20, i, str(total), ha='left', va='center', fontsize=28, color='gray')

# Add legend and labels
plt.xlabel('Number of Segments', fontsize=32)
# plt.ylabel('Bird Species', fontsize=16)
# plt.title('Stacked Number of Segments by Bird Species', fontsize=26)
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.gca().patch.set_facecolor('none')
plt.gca().invert_yaxis()
plt.ylim(-0.3, len(df) - 0.3)

# Set legend background to transparent and adjust position
legend = plt.legend(title='Regions', fontsize=26, title_fontsize=28, loc='center', bbox_to_anchor=(0.8, 0.5))
frame = legend.get_frame()
frame.set_alpha(1.0)
frame.set_facecolor('white')
frame.set_linewidth(1)
frame.set_edgecolor('gray')

# Remove borders
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# Set the figure background to transparent
plt.gcf().patch.set_alpha(0.0)

# Display the figure
plt.show()



###################################### Stacked Bar Chart of Number of Segments by Sound Type and Bird Species ######################################

# Data preparation
data = {
    'Bird Species': [f'#{i}' for i in range(10)],
    'Common Name': ['Red-winged Blackbird', 'Northern Cardinal', 'Brown Creeper',
                    'American Crow', 'Brown-headed Cowbird', 'American Yellow Warbler',
                    'American Redstart', 'American Goldfinch', 'Willet', 'American Robin'],
    'Sound Type': ['Songs', 'Songs', 'Calls', 'Calls', 'Calls', 'Songs', 'Songs', 'Songs', 'Calls', 'Songs'],
    'Number of Segments': [2188, 2243, 489, 1203, 538, 1036, 885, 598, 273, 2016],
    'frequency range (kHz)': ['2.8-5.7', '3.5-4.0', '3.7-8.0', '0.5-1.8', '0.5-12.0', '3.0-8.0',
                              '3.0-8.0', '1.6-6.7', '1.5-2.5', '1.8-3.7']
}

df = pd.DataFrame(data)

# Custom colors
colors = [
    (153/255, 229/255, 255/255),  # Agelaius phoeniceus
    (242/255, 170/255, 132/255),  # Cardinalis cardinalis
    (35/255, 61/255, 220/255),     # Certhia americana
    (25/255, 101/255, 41/255),     # Corvus brachyrhynchos
    (255/255, 140/255, 0/255),     # Molothrus ater
    (98/255, 192/255, 171/255),    # Setophaga aestiva
    (115/255, 237/255, 0/255),     # Setophaga ruticilla
    (101/255, 103/255, 50/255),    # Spinus tristis
    (228/255, 0/255, 232/255),     # Tringa semipalmata
    (254/255, 255/255, 0/255)      # Turdus migratorius
]

# Function to set font sizes
def set_fontsize():
    plt.title(plt.gca().get_title(), fontsize=18)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=16)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

# Stacked Bar Chart of Number of Segments by Sound Type and Bird Species
df_grouped = df.groupby(['Sound Type', 'Bird Species'])['Number of Segments'].sum().unstack()
plt.figure(figsize=(10, 6), facecolor='none')
ax = df_grouped.plot(kind='bar', stacked=True, color=colors, colormap='coolwarm', ax=plt.gca(), legend=False)
# plt.title('Number of Segments by Sound Type', fontsize=34)
plt.xlabel('Sound Type', fontsize=30)
plt.ylabel('Number of Segments', fontsize=32)
plt.xticks(rotation=0, fontsize=26)
plt.yticks(fontsize=28)
plt.grid(False)
plt.gca().patch.set_facecolor('none')  # Set background to transparent
# Remove borders
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

################################################# Frequency Range by Bird Species ######################################
plt.figure(figsize=(5, 7), facecolor='none')
handles = []  # To store legend handles

plt.rcParams['font.family'] = 'Times New Roman'
for index, row in df.iterrows():
    freqs = list(map(float, row['frequency range (kHz)'].split('-')))
    line, = plt.plot([freqs[0], freqs[1]], [index, index], marker='o', color=colors[index], linewidth=20)  # Set line width
    # Add legend
    handles.append((line, row['frequency range (kHz)']))

plt.yticks(range(len(df['Bird Species'])), df['Bird Species'], fontsize=28)
plt.xlabel('Frequency Range (kHz)', fontsize=28)
plt.xticks(fontsize=27)
# plt.title('Frequency Range of Bird Species', fontsize=20)
# set_fontsize()
plt.gca().patch.set_facecolor('none')  # Set background to transparent

# Add legend and adjust position
legend = plt.legend(*zip(*handles), title='Frequency', fontsize=24, title_fontsize=28,
                    loc='upper left', bbox_to_anchor=(1.05, 1), frameon=True, handlelength=0.2)
legend.get_frame().set_facecolor('none')  # Set legend background to transparent
legend.get_frame().set_edgecolor('gray')  # Set legend border color

plt.gca().invert_yaxis()  # Invert y-axis
# Remove borders
for spine in plt.gca().spines.values():
    spine.set_visible(False)
plt.show()

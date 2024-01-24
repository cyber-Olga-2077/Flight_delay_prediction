import pandas as pd
import math

def calculate_dist_between_airports(airport1_code, airport2_code):
    def calc_distance(x1, y1, x2, y2):
        distance = math.sqrt(math.pow((x2 - x1), 2) + math.pow((math.cos((x1 * math.pi) / 180) * (y2 - y1)), 2)) * (40075.704 / 360)
        distance = round(distance, 2)
        return distance

    airports = pd.read_csv('https://davidmegginson.github.io/ourairports-data/airports.csv', sep=',')
    del airports['home_link'], airports['iso_region'], airports['continent'], airports['wikipedia_link'], airports['elevation_ft'], airports['scheduled_service'], airports['gps_code'], airports['local_code'], airports['keywords']
    airports = airports.loc[airports.iata_code.notnull()]  # Remove airports without IATA code

    try:
        airport1_data = airports.loc[airports['iata_code'] == airport1_code, ['name', 'latitude_deg', 'longitude_deg']].iloc[0]
        airport2_data = airports.loc[airports['iata_code'] == airport2_code, ['name', 'latitude_deg', 'longitude_deg']].iloc[0]

        distance = calc_distance(airport1_data['latitude_deg'], airport1_data['longitude_deg'], airport2_data['latitude_deg'], airport2_data['longitude_deg'])
        distance = round(distance, 2)
        return distance
    except IndexError:
        print("Please enter correct IATA codes")
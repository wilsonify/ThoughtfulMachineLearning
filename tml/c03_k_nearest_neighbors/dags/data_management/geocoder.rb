require 'csv'
require 'dalli'
require 'geocoder'
require 'pry'

# Configure Bing Geocoder (API key required)
Geocoder.configure(lookup: :bing, api_key: "")

# Initialize Memcached client
dc = Dalli::Client.new('localhost:11211')

headers = nil
count   = 0

IGNORE_HEADERS = %w[
  id TaxableValue Address ErosionHazard LandfillBuffer HundredYrFloodPlain
  SeismicHazard LandslideHazard SteepSlopeHazard Stream Wetland SpeciesOfConcern
  SensitiveAreaTract AirportNoise DNRLease CommonProperty CoalMineHazard CriticalDrainage
]

# Open output CSV
CSV.open('./king_county_data_geocoded.csv', 'wb') do |csv|
  CSV.foreach('/tmp/king_county_data.csv', headers: true) do |row|
    # Build and write headers once
    headers ||= begin
      filtered_headers = row.headers.reject { |h| IGNORE_HEADERS.include?(h) }
      headers = filtered_headers + ['lat', 'long']
      csv << headers
      headers
    end

    # Function to map 'Y'/'N' values to 1/0
    binary = ->(x) do
      case x
      when 'N' then 0
      when 'Y' then 1
      else x
      end
    end

    # Only geocode every 9th record
    if count % 9 == 0
      response = dc.get(row['id'])

      unless response
        sleep 0.25 # throttle requests
        geocode = Geocoder.search(row['Address'])
        if geocode.empty?
          puts "Nothing back from Bing for #{row['id']} (#{row['Address']})"
          count += 1
          next
        end
        response = geocode.first.coordinates
        dc.set(row['id'], response)
      end

      puts "Processed row #{count}"

      lat, long = response
      appraised_value = row['AppraisedValue'].to_f

      if appraised_value < 1_000_000 && appraised_value > 10_000
        r = []
        row.to_hash.each do |k, v|
          next if IGNORE_HEADERS.include?(k)
          r << binary.call(v)
        end
        r += [lat, long]
        csv << r
      end
    end

    count += 1
  end
end

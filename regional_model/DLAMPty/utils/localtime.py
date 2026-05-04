import numpy as np
from datetime import datetime, timedelta

# Example UTC time
utc_time = datetime.utcnow()  # Get the current UTC time
# Alternatively, you can specify a specific UTC time
# utc_time = datetime(2023, 10, 1, 12, 0, 0)  # Example: October 1, 2023, 12:00 PM UTC

# Example 2D array of longitudes (0 to 360)
longitudes = np.array([[285.0, 0.0], [30.0, 120.0]])  # Example longitudes

# Convert longitudes from 0-360 to -180 to 180
adjusted_longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)

# Calculate the offset in hours
offset_hours = adjusted_longitudes / 15.0  # Calculate the offset in hours

# Convert UTC time to a timestamp
utc_timestamp = utc_time.timestamp()

# Calculate local times by adding the offset (in seconds) to the UTC timestamp
local_timestamps = utc_timestamp + (offset_hours * 3600)

# Convert back to datetime objects
local_times = np.array([datetime.fromtimestamp(ts) for ts in local_timestamps.flatten()]).reshape(local_timestamps.shape)

# Print results
print("Local Times:")
print(local_times)

# Data items in the 2025 challenge
April 2025

Each item was collected using participants' Android smartphones or smartwatches, with data sampling intervals ranging from 1 to 10 minutes. It should be noted that certain data items may contain substantial levels of missing data and noise. Participants may not have consistently worn or carried their smartphones or smartwatches throughout the full 24-hour period. Furthermore, sensor measurements and recordings could have been interrupted due to device charging or system reboots.

The detailed structure of individual sensor data items is summarized in Table 1. Each data item is stored as an individual data file and is provided along with the participant ID and timestamp. All timestamps are recorded based on **Korea Standard Time (KST)** and are displayed in the **YYYY-MM-DD HH:MM:SS** format (e.g., 2024-08-01 12:34:56). To protect privacy, certain data items have undergone minimal anonymization. For example, GPS latitude and longitude data are provided as relative coordinates.

### Table 1: The detailed composition

| Items | Column | Data type | Note |
|---|---|---|---|
| mACStatus | m_charging | integer | 0: No, 1: Charging |
| mActivity | m_activity | integer | 0: in_vehicle, 1: on_bicycle, 2: on_foot, 3: still, 4: unknown, 5: tilting, 7: walking, 8: running |
| mAmbience | m_ambience | object | List of ambient sound labels and their respective probabilities |
| mBle | m_ble | object | List of bluetooth device address, device_class, and rssi |
| mGps | m_gps | object | List of (altitude, latitude, longitude, speed) |
| mLight | m_light | float | Ambient light in lx unit |
| mScreenStatus | m_screen_use | integer | 0: No, 1: Using screen |
| mUsageStats | m_usage_stats | object | List of app names and their respective usage times (in milliseconds unit) |
| mWifi | m_wifi | object | List of base station ID(bssid) and rssi |
| wHr | heart_rate | object | List of heart rate recordings |
| wLight | w_light | float | Ambient light in lx unit |
| wPedo | burned_calories<br>distance<br>speed<br>step<br>step_frequency | float<br>float<br>float<br>integer<br>float | Number of calories<br>Distance in meters<br>Speed in km/h unit<br>Number of steps<br>Step frequency in a minute |

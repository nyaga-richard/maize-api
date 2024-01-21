# Input list with duplicates
counties = [
    'Baringo Central', 'Baringo North', 'Baringo South', 'East Pokot', 'Mogotio', 'Eldama Ravine',
'Bomet Central', 'Bomet East', 'Chepalangu', 'Konoin', 'Sotik',
'Bumula', 'Kabuchai', 'Kanduyi', 'Kimilili', 'Mt.Elgon', 'Sirisia', 'Tongaren', 'Webuye East', 'Webuye West', 'Samia',
'Bunyala', 'Butula', 'Matayos', 'Teso South', 'Nambale', 'Teso North', 'Keiyo North', 'Keiyo South',
'Marakwet East', 'Marakwet West', 'Manyatta', 'Mbeere North', 'Mbeere South', 'Runyenjes',
'Garissa', 'Fafi', 'Balambala', 'Dadaab', 'Lagdera', 'Ijara', 'Hulugho',
'Suba', 'Kabondo', 'Karachuonyo', 'Kasipul', 'Ndhiwa', 'Rangwe', 'Mbita', 'Homa Bay Town',
'Merti', 'Isiolo', 'Garba Tulla', 'Kajiado Central', 'Kajiado East', 'Kajiado North',
'Kajiado South', 'Kajiado West', 'Khwisero', 'Likuyani', 'Butere', 'Lugari', 'Malava',
'Matete', 'Mumias East', 'Mumias West', 'Shinyalu', 'Lurambi', 'Navakholo', 'Ikolomani', 'Matungu',
'Ainamoi', 'Belgut', 'Bureti', 'Kipkelion East', 'Kipkelion West', 'Soin-Sigowet', 'Kabete', 'Thika',
'Kiambu', 'Gatundu South', 'Kikuyu', 'Juja', 'Limuru', 'Lari', 'Ruiru', 'Kiambaa', 'Gatundu North',
'Githunguri', 'Kaloleni', 'Ganze', 'Kilifi North', 'Magarini', 'Malindi', 'Rabai', 'Kilifi South',
'Kirinyaga West', 'Kirinyaga Central', 'Mwea East', 'Mwea West', 'Kirinyaga East', 'Kitutu Chache North',
'South Mugirango', 'Boma Choge Borabu', 'Nyaribari Chache', 'Bobasi', 'Nyaribari Masaba', 'Kitutu Chache South',
'Boma Choge Chache', 'Bonchari', 'Nyakach', 'Seme', 'Kisumu West', 'Muhoroni', 'Nyando', 'Kisumu East',
'Kitui Central', 'Kitui East', 'Kitui Rural', 'Kitui South', 'Kitui West', 'Mwingi Central', 'Mwingi North',
'Mwingi West', 'Matuga', 'Lungalunga', 'Msambweni', 'Kinango', 'Laikipia East', 'Laikipia West', 'Laikipia North',
'Lamu East', 'Lamu West', 'Machakos', 'Mwala', 'Kathiani', 'Kangundo', 'Matungulu', 'Mavoko', 'Masinga', 'Yatta',
'Makueni', 'Kilome', 'Mbooni', 'Kaiti', 'Kibwezi East', 'Kibwezi West', 'Laisamis', 'Moyale', 'North Horr', 'Saku',
'Tigania Central', 'Imenti South', 'Buuri', 'Tigania West', 'Igembe North', 'Imenti Central', 'Imenti North',
'Tigania East', 'Igembe Central', 'Igembe South', 'Rongo', 'Suna East', 'Suna West', 'Kuria East', 'Awendo',
'Kuria West', 'Nyatike', 'Uriri', 'Nairobi', 'Kisauni', 'Changamwe', 'Jomvu', 'Mvita', 'Likoni', 'Gatanga',
'Kahuro', 'Kigumo', 'Kangema', 'Kandara', 'Maragua', 'Mathioya', 'Kiharu', 'Roysambu', 'Ruaraka', 'Kasarani',
'Embakasi South', 'Embakasi Central', 'Embakasi East', 'Embakasi West', 'Embakasi North', 'Kibra', 'Langata',
'Dagoretti South', 'Dagoretti North', 'Kamukunji', 'Makadara', 'Westlands', 'Starehe', 'Bahati', 'Nakuru West',
'Gilgil', 'Kuresoi North', 'Kuresoi South', 'Molo', 'Nakuru East', 'Naivasha', 'Njoro', 'Rongai', 'Subukia',
'Emgwen', 'Chesumei', 'Mosop', 'Tinderet', 'Aldai', 'Nandi Hills', 'Narok South', 'Narok West', 'Transmara West',
'Transmara East', 'Narok East', 'Narok North', 'Nyamira South', 'Borabu', 'Masaba North', 'Manga', 'Nyamira North',
'Nyamira South', 'Kipipiri', 'Olkalau', 'Kinangop', 'Ndaragwa', 'Oljororok', 'Mathira West', 'Mathira East',
'Kieni East', 'Kieni West', 'Mukurweini', 'Nyeri South', 'Tetu', 'Nyeri Central', 'Samburu Central', 'Samburu North',
'Samburu East', 'Alego Usonga', 'Rarieda', 'Ugenya', 'Gem', 'Ugunja', 'Bondo', 'Voi', 'Taveta', 'Mwatate', 'Taita',
'Tana River', 'Tana North', 'Tana Delta', 'Tharaka South', 'Tharaka North', 'Meru South', 'Maara', 'Endebess',
'Cheranganyi', 'Kiminini', 'Kwanza', 'Saboti', 'Turkana East', 'Turkana West', 'North/Kibish', 'Loima', 'Turkana South',
'Turkana Central', 'Turbo', 'Kapseret', 'Ainabkoi', 'Moiben', 'Kesses', 'Soy', 'Sabatia', 'Hamisi', 'Emuhaya', 'Luanda',
'Wajir East', 'Tarbaj', 'Wajir West', 'Eldas', 'Wajir North', 'Bute', 'Wajir South', 'Habaswein', 'West Pokot',
'Pokot North', 'Pokot Central', 'Pokot South'

#    'BARINGO', 'BARINGO', 'BARINGO', 'BARINGO', 'BARINGO', 'BARINGO',
# 'BOMET', 'BOMET', 'BOMET', 'BOMET', 'BOMET',
# 'BUNGOMA', 'BUNGOMA', 'BUNGOMA', 'BUNGOMA', 'BUNGOMA', 'BUNGOMA', 'BUNGOMA', 'BUNGOMA', 'BUNGOMA',
# 'BUSIA', 'BUSIA', 'BUSIA', 'BUSIA', 'BUSIA', 'BUSIA', 'BUSIA',
# 'ELGEYO MARAKWET', 'ELGEYO MARAKWET', 'ELGEYO MARAKWET', 'ELGEYO MARAKWET',
# 'EMBU', 'EMBU', 'EMBU', 'EMBU',
# 'GARISSA', 'GARISSA', 'GARISSA', 'GARISSA', 'GARISSA', 'GARISSA', 'GARISSA',
# 'HOMABAY', 'HOMABAY', 'HOMABAY', 'HOMABAY', 'HOMABAY', 'HOMABAY', 'HOMABAY', 'HOMABAY',
# 'ISIOLO', 'ISIOLO', 'ISIOLO',
# 'KAJIADO', 'KAJIADO', 'KAJIADO', 'KAJIADO', 'KAJIADO',
# 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA', 'KAKAMEGA',
# 'KERICHO', 'KERICHO', 'KERICHO', 'KERICHO', 'KERICHO', 'KERICHO',
# 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU', 'KIAMBU',
# 'KILIFI', 'KILIFI', 'KILIFI', 'KILIFI', 'KILIFI', 'KILIFI', 'KILIFI',
# 'KIRINYAGA', 'KIRINYAGA', 'KIRINYAGA', 'KIRINYAGA', 'KIRINYAGA',
# 'KISII', 'KISII', 'KISII', 'KISII', 'KISII', 'KISII', 'KISII', 'KISII', 'KISII',
# 'KISUMU', 'KISUMU', 'KISUMU', 'KISUMU', 'KISUMU', 'KISUMU',
# 'KITUI', 'KITUI', 'KITUI', 'KITUI', 'KITUI', 'KITUI', 'KITUI', 'KITUI',
# 'KWALE', 'KWALE', 'KWALE', 'KWALE',
# 'LAIKIPIA', 'LAIKIPIA', 'LAIKIPIA',
# 'LAMU', 'LAMU',
# 'MACHAKOS', 'MACHAKOS', 'MACHAKOS', 'MACHAKOS', 'MACHAKOS', 'MACHAKOS', 'MACHAKOS', 'MACHAKOS',
# 'MAKUENI', 'MAKUENI', 'MAKUENI', 'MAKUENI', 'MAKUENI', 'MAKUENI',
# 'MARSABIT', 'MARSABIT', 'MARSABIT', 'MARSABIT',
# 'MERU', 'MERU', 'MERU', 'MERU', 'MERU', 'MERU', 'MERU', 'MERU', 'MERU', 'MERU',
# 'MIGORI', 'MIGORI', 'MIGORI', 'MIGORI', 'MIGORI', 'MIGORI', 'MIGORI', 'MIGORI',
# 'MOMBASA', 'MOMBASA', 'MOMBASA', 'MOMBASA', 'MOMBASA', 'MOMBASA',
# 'MURANGA', 'MURANGA', 'MURANGA', 'MURANGA', 'MURANGA', 'MURANGA', 'MURANGA', 'MURANGA',
# 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI', 'NAIROBI',
# 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU', 'NAKURU',
# 'NANDI', 'NANDI', 'NANDI', 'NANDI', 'NANDI',
# 'NAROK', 'NAROK', 'NAROK', 'NAROK', 'NAROK', 'NAROK',
# 'NYAMIRA', 'NYAMIRA', 'NYAMIRA', 'NYAMIRA', 'NYAMIRA', 'NYAMIRA',
# 'NYANDARUA', 'NYANDARUA', 'NYANDARUA', 'NYANDARUA', 'NYANDARUA',
# 'NYERI', 'NYERI', 'NYERI', 'NYERI', 'NYERI', 'NYERI', 'NYERI', 'NYERI',
# 'SAMBURU', 'SAMBURU', 'SAMBURU',
# 'SIAYA', 'SIAYA', 'SIAYA', 'SIAYA', 'SIAYA', 'SIAYA',
# 'TAITA TAVETA', 'TAITA TAVETA', 'TAITA TAVETA', 'TAITA TAVETA',
# 'TANA RIVER', 'TANA RIVER', 'TANA RIVER',
# 'THARAKA NITHI', 'THARAKA NITHI', 'THARAKA NITHI', 'THARAKA NITHI',
# 'TRANS NZOIA', 'TRANS NZOIA', 'TRANS NZOIA', 'TRANS NZOIA', 'TRANS NZOIA',
# 'TURKANA', 'TURKANA', 'TURKANA', 'TURKANA', 'TURKANA', 'TURKANA',
# 'UASIN GISHU', 'UASIN GISHU', 'UASIN GISHU', 'UASIN GISHU', 'UASIN GISHU', 'UASIN GISHU',
# 'VIHIGA', 'VIHIGA', 'VIHIGA', 'VIHIGA',
# 'WAJIR', 'WAJIR', 'WAJIR', 'WAJIR', 'WAJIR', 'WAJIR', 'WAJIR', 'WAJIR',
# 'WEST POKOT', 'WEST POKOT', 'WEST POKOT', 'WEST POKOT'

]

# Remove duplicates
unique_counties = list(set(counties))

# Convert to string array format
string_array_format = [
    f'<item>{county}</item>' for county in unique_counties
]

# Print the formatted string array
print('<string-array name="planets_array">')
print('\n'.join(string_array_format))
print('</string-array>')

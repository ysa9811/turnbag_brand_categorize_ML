import requests
import pymysql
import requests
from io import BytesIO
import pandas as pd
import matplotlib.image as mpimg
from concurrent.futures import ThreadPoolExecutor

MYSQL_CONFIG = {
    "host": "43.203.182.106",
    "port": 3306,
    "user": "turnbag_crawler",
    "password": "!@crawler#$2024",
    "database": "crawler_db",
    "muser": "turnbag_modeling",
    "mpassword": "!@modeling#$2024",
    "mdatabase": "modeling_db",
}


def connect_to_mysql():
    connection = pymysql.connect(
        host=MYSQL_CONFIG["host"],
        port=MYSQL_CONFIG["port"],
        user=MYSQL_CONFIG["muser"],
        password=MYSQL_CONFIG["mpassword"],
        database=MYSQL_CONFIG["mdatabase"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )
    return connection


def BALENCIAGA():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'BALENCIAGA'"
        "group by g.idx"
    )
    return sql

def BOTTEGAVENETA():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'BOTTEGA VENETA'"
        "group by g.idx LIMIT 5437"
    )
    return sql

def BURBERRY():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'BURBERRY'"
        "group by g.idx"
    )
    return sql

def CELINE():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'CELINE'"
    )
    return sql

def Citizen():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Citizen'"
    )
    return sql

def DIOR():
    # sql = (
    #     "SELECT gi.idx, g.brand_name, gi.image_url "
    #     "FROM goods_image gi, goods g "
    #     "WHERE gi.goods_idx = g.idx "
    #     "AND g.brand_name = 'DIOR'"
    #     "group by g.idx"
    # )
    sql = (
        "WITH RankedImages AS ("
        "SELECT gi.idx, g.brand_name, gi.image_url, g.idx AS g_idx, "
        "ROW_NUMBER() OVER (PARTITION BY g.idx ORDER BY gi.idx) AS row_num "
        "FROM goods_image gi "
        "JOIN goods g ON gi.goods_idx = g.idx "
        "WHERE g.brand_name = 'DIOR') "
        "SELECT idx, brand_name, image_url, g_idx "
        "FROM RankedImages WHERE row_num <= 2;"
    )
    return sql

def Fendi():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Fendi'"
    )
    return sql

def GUCCI():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'GUCCI'"
        "and gi.idx not in(146198, 146212, 146185, 146184, 146182, 146181, 146180, 146179, "
        "146166, 146117, 146116, 146115, 146114, 146113, 146112, 146111, 146110, 146109, 146108, "
        "146036, 145995, 145942, 145941, 145940, 145939, 145932, 145931, 145930, 145902, 145800, 145434, "
        "145328, 145327, 145326, 145325, 145324, 145323, 145316, 145198, 17875, 17774, 17748, 17549, 17506)"
        "group by g.idx"
    )
    return sql

def Hamilton():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Hamilton'"
        "and gi.idx not in(18124, 18127, 18123, 18120, 18117, 18114, 18111, 18108, 18097, 18100, "
        "18093, 18090, 18087, 18084, 18081, 18059, 18056, 18053, 18050, 18047, 18043, 18042, "
        "18039, 18038, 18035, 18032, 18029, 18026, 18022, 18021, 18018, 18017, 18014, 18011, "
        "18008, 18004, 17988, 17987, 17984, 17983, 17980, 17977, 17974, 17937, 17936, 17970, 17933, "
        "17932, 17929, 17926, 17923, 17919, 17903, 17855, 17899, 17851, 17798)"
    )
    return sql

def HERMES():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'HERMES' and gi.idx "
        "in(39606, 39612, 39615, 39618, 39623, 39624, 39627, 39642, 39687, "
        "39688, 39962, 40014, 40035, 40047, 40065, 40196, 40254, 40450, 40452, 40459, 40458)"
    )
    return sql

def Longines():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Longines' and gi.idx "
        "not in(55969, 55971, 55980, 55982, 55988, 56008, 56011, 56064, 56078, 56082, 56086, 56090, 56098, 56118, 56129, 62463)"
    )
    return sql

def LouisVuitton():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Louis Vuitton' "
        "and gi.idx not in(39498, 39497, 38912, 38523, 38522, 38511, 38510, 38508, 38501, 38495, 38398, 37879, "
        "34750, 34749, 34685, 34684, 34652, 34651, 32759, 32758, 32323, 32322, 32318, 32317, 31304, "
        "31303, 30866, 30865, 30846, 30806, 30461, 30460, 29936) group by g.idx "
    )
    return sql

def MAISON246():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'MAISON246'"
        "group by g.idx "
    )
    return sql

def MichaelKors():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Michael Kors'"
    )
    return sql

def MONCLER():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'MONCLER'"
    )
    return sql

def Rolex():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Rolex' "
        "and gi.idx in (148263, 142867, 148271, 148275, 148279, 148283, 148416, "
        "148420, 148424, 148452, 148456, 148460, 148464, 148468, 148472)"
    )
    return sql

def ROMANSON():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'ROMANSON' "
        "and gi.idx not in (18573, 18574, 18575, 18576, 18577, 18578, 18579, 18580, 18581, 18582, 18583,  "
        "18584, 18594, 18595, 18596, 18597, 18598, 18599, 18600, 18601, 18602, 18603, 18604, 18605, 18623, 18624,"
        "18625, 18626, 18627, 18628, 18629, 18630, 18631, 18632, 18633, 18634, 18676, 18677, 18678, 18697, 18698, "
        "18699, 18724, 18725, 18726, 18915, 18916, 18917, 18942, 18943, 18944, 18980, 18981, 18982, 19003, 19004, 19005, "
        "19033, 19034, 19035) group by g.idx"
    )
    return sql

def THOMBROWNE():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'THOM BROWNE' "
        "and gi.idx not in (2284, 2288, 2289, 2291, 2296, 2298, 2302, 2304, 2310, 2312, 2317, 2324, 2325, 2358, 2361, "
        "2366, 2368, 2377, 2378, 2379, 2386, 2389, 2393, 2395, 2396, 2400, 2407, 2414, 2416, 2417, 2420, 2423, 2427, 2435, "
        "2437, 2443, 2445, 2450, 2457, 2458, 2467, 2468, 2470, 2479, 2485, 2487, 2493, 2495, 2499, 2501, 2506, 2509, 2513, "
        "2515, 2516, 2520, 2521, 2526, 2533, 2540, 2547, 2552, 2555, 2562, 2568, 2576, 2585, 2594, 2615, 2620, 2628, 2629, "
        "2630, 2631, 2639, 2642, 2649, 2656, 2663, 2669, 2671, 2676, 2679, 2688, 2695, 2702, 2710, 2719, 2726, 2735, "
        "2740, 2743, 2752, 2759, 2766, 2773, 2779, 2782, 2790, 2798, 2806, 2812, 2814, 2819, 2822, 2827, 2831, 2840, 2848, "
        "2856, 2865, 2873, 2876, 2880, 2885, 2889, 2894, 2899, 2908, 2918, 2921, 2925, 2929, 2938, 2947, 2956, 2961, "
        "2963, 2971, 2975, 2976, 2977, 2978, 2981, 2989, 2995, 2998, 3003, 3004, 3005, 3006, 3009, 3014, 3018, 3026, "
        "3036, 3040, 3041, 3042)"
    )
    return sql

def SALVATOREFERRAGAMO():
    sql = (
        "WITH RankedImages AS ( "
        "SELECT gi.idx, g.brand_name, gi.image_url, g.idx AS g_idx, "
        "ROW_NUMBER() OVER (PARTITION BY g.idx ORDER BY gi.idx) AS row_num "
        "FROM goods_image gi "
        "JOIN goods g ON gi.goods_idx = g.idx "
        "WHERE g.brand_name = 'SALVATORE FERRAGAMO ') "
        "SELECT idx, brand_name, image_url, g_idx "
        "FROM RankedImages WHERE row_num <= 2;"
    )
    # sql = (
    #     "SELECT gi.idx, g.brand_name, gi.image_url "
    #     "FROM goods_image gi, goods g "
    #     "WHERE gi.goods_idx = g.idx "
    #     "AND g.brand_name = 'SALVATORE FERRAGAMO' "
    #     "and gi.idx not in (12325, 12330, 12339, 12343, 12348, 12358, 12363, 12369, 12375, 12381, 12388, 12393, 12406, "
    #     "12411, 12416, 12421, 12426, 12431, 12435, 12440, 12455, 12460, 12465, 12469, 12474, 12565, 12569, "
    #     "12573, 12577, 12581, 12592, 12596, 12600, 12604, 12608, 12612, 12616, 12620, 12624, 12642, 12648) group by g.idx  "
    # )
    return sql

def PRADA():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'PRADA' "
        "group by g.idx "
    )
    return sql


def Chanel1():
    sql = (
        "SELECT gi.idx, g.brand_name, gi.image_url "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "AND g.brand_name = 'Chanel' "
        "and gi.idx not in(41139, 41138, 41137, 41100, 41074, 41050, 41002, 40987, 40955, 40917, 40895, 40929, 40928, 40894, 40893, 40869, 40861, 40836, 40756, 40755, 40496);"
    )
    return sql

def Chanel2():
    sql = (
        "select gi.idx, g.brand_name, gi.image_url, g.idx as g_idx "
        "FROM goods_image gi, goods g "
        "WHERE gi.goods_idx = g.idx "
        "and g.brand_name = '샤넬' AND gi.idx NOT BETWEEN 23244 AND 34514 "
        "and gi.idx not in(16406, 16420, 16423, 16435, 16442, 16449, 16459, 16465, 16471, 16524) group by g.idx LIMIT 800"
    )
    return sql

def mysql():
    connection = connect_to_mysql()
    try:
        with connection.cursor() as cursor:
            # 여러 SQL 쿼리를 호출하여 결과 합치기
            sql_queries = [
                BALENCIAGA(),
                BOTTEGAVENETA(),
                BURBERRY(),
                CELINE(),
                Citizen(),
                DIOR(),
                Fendi(),
                GUCCI(),
                Hamilton(), 
                HERMES(), 
                Longines(), 
                LouisVuitton(), 
                MAISON246(), 
                MichaelKors(), 
                MONCLER(), 
                Rolex(), 
                ROMANSON(), 
                THOMBROWNE(), 
                SALVATOREFERRAGAMO(), 
                PRADA(), 
                Chanel1(),
                Chanel2()
            ]
            
            # 모든 쿼리 결과를 합침
            all_targets = []
            for sql in sql_queries:
                cursor.execute(sql)
                targets = cursor.fetchall()
                all_targets.extend(targets)  # 쿼리 결과를 all_targets에 추가

            print(f"Total targets: {len(all_targets)}")
            image_all_data = url_image_data(all_targets)
        
    finally:
        connection.close()

    return image_all_data



# url rgb화화
def fetch_image(target):
    url = target["image_url"]
    if not url.startswith(("http://", "https://")):
        return None

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            img_data = BytesIO(response.content)
            image = mpimg.imread(img_data, format="jpeg")
            return target["idx"], target["brand_name"], image
    except Exception:
        return None


# pd데이터로 변환환
def url_image_data(targets):
    with ThreadPoolExecutor() as executor:
        results = executor.map(fetch_image, targets)

    data = [res for res in results if res]
    img_all_data = pd.DataFrame(data, columns=["idx", "brand_name", "image_rgb"])
    brand_mapping = {
        'BALENCIAGA': 'BALENCIAGA',
        'Balenciaga' : 'BALENCIAGA',
        'BOTTEGA VENETA': 'BOTTEGA VENETA',
        'BURBERRY': 'BURBERRY',
        'Burberry': 'BURBERRY',
        'CELINE': 'CELINE',
        'Citizen': 'Citizen',
        'DIOR': 'DIOR',
        'Fendi': 'Fendi',
        'GUCCI': 'GUCCI',
        'Gucci': 'GUCCI',
        'Hamilton': 'Hamilton',
        'HERMES': 'HERMES',
        'Longines': 'Longines',
        'Louis Vuitton': 'Louis Vuitton',
        'MAISON246': 'MAISON246',
        'Michael Kors': 'Michael Kors',
        'MONCLER': 'MONCLER',
        'Rolex': 'Rolex',
        'ROMANSON': 'ROMANSON',
        'THOM BROWNE': 'THOM BROWNE',
        'SALVATORE FERRAGAMO': 'SALVATORE FERRAGAMO',
        'PRADA': 'PRADA',
        'Chanel': 'Chanel',
        '샤넬': 'Chanel'
    }
    img_all_data['brand_name'] = img_all_data['brand_name'].replace(brand_mapping)
    print(img_all_data['brand_name'].unique())
    brand_counts = img_all_data['brand_name'].value_counts()
    print("브랜드별 데이터 개수:\n", brand_counts)
    categories = ['BALENCIAGA', 'BOTTEGA VENETA', 'BURBERRY', 'CELINE', 'Citizen', 'DIOR', 'Fendi', 'GUCCI', 
                  'Hamilton', 'HERMES', 'Longines', 'Louis Vuitton', 'MAISON246', 'Michael Kors', 'MONCLER', 
                  'Rolex', 'ROMANSON', 'THOM BROWNE', 'SALVATORE FERRAGAMO', 'PRADA', 'Chanel']
    img_all_data['brand_name'] = pd.Categorical(
        img_all_data['brand_name'], categories=categories, ordered=True
    ).codes
    print("\n==================================\n")
    return img_all_data

import requests
import pymysql
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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


def mysql():
    connection = connect_to_mysql()
    try:
        with connection.cursor() as cursor:
            # 상태 업데이트 SQL 쿼리 작성
            #sql = "SELECT idx, image_url FROM goods_image;"
            sql = (
                "select gi.idx, g.brand_name, gi.image_url, g.idx as g_idx "
                "FROM goods_image gi, goods g "
                "WHERE gi.goods_idx = g.idx "
                "and g.brand_name = '샤넬' AND gi.idx NOT BETWEEN 23244 AND 34514 group by g.idx order by gi.idx asc LIMIT 800"
            )
            cursor.execute(sql)
            targets = cursor.fetchall()  # idx와 image_url 포함된 리스트 반환
    finally:
        connection.close()

    return [{'idx': target['idx'], 'image_url': target['image_url']} for target in targets]




def fetch_images_and_indices(targets, start_idx, count):
    images = []
    indices = []
    for target in targets[start_idx:start_idx + count]:
        try:
            response = requests.get(target['image_url'])
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            images.append(img)
            indices.append(target['idx'])
        except Exception:
            images.append(None)  # 오류 시 빈 이미지 추가
            indices.append(target['idx'])  # idx는 유지
    return images, indices


def plot_images(images, indices, page, total_pages):
    global fig, axes, btn_next, btn_prev
    if fig is None or axes is None:
        fig, axes = plt.subplots(4, 5, figsize=(15, 6))
        axes = axes.flatten()

        ax_next = plt.axes([0.8, 0.02, 0.1, 0.05])
        ax_prev = plt.axes([0.1, 0.02, 0.1, 0.05])
        btn_next = Button(ax_next, 'Next')
        btn_prev = Button(ax_prev, 'Previous')

        btn_next.on_clicked(lambda event: next_page(event, targets))
        btn_prev.on_clicked(lambda event: prev_page(event, targets))

    for i, ax in enumerate(axes):
        ax.clear()
        if i < len(images) and images[i]:
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(f"idx: {indices[i]}", fontsize=10)
        else:
            ax.axis('off')

    fig.suptitle(f"Page {page + 1} of {total_pages}")
    plt.tight_layout()
    plt.draw()  # 플롯 갱신

def next_page(event, targets):
    global current_page
    if (current_page + 1) * 20 < len(targets):
        current_page += 1
        render_page(targets)

def prev_page(event, targets):
    global current_page
    if current_page > 0:
        current_page -= 1
        render_page(targets)

def render_page(targets):
    total_pages = (len(targets) + 19) // 20
    start_idx = current_page * 20
    images, indices = fetch_images_and_indices(targets, start_idx, 20)
    plot_images(images, indices, current_page, total_pages)


fig, axes, btn_next, btn_prev = None, None, None, None
current_page = 39

targets = mysql()
render_page(targets)

plt.show()
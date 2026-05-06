"""
Bing 图片爬虫（国内可用版）
用法：python spider.py --keyword "猫" --num 30
"""

import os
import re
import requests
import argparse
import time
import random
from tqdm import tqdm

def download_bing_images(keyword, save_dir, num_images=30):
    """从 Bing 图片下载图片"""
    os.makedirs(save_dir, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'zh-CN,zh;q=0.9',
        'Referer': 'https://www.bing.com/',
    }
    
    downloaded = 0
    offset = 0
    
    print(f"开始搜索关键词: {keyword}")
    
    while downloaded < num_images:
        # Bing 图片搜索 API
        url = f'https://www.bing.com/images/search?q={keyword}&first={offset}&count=35'
        
        try:
            response = requests.get(url, headers=headers, timeout=15)
            
            # 提取图片 URL（多种正则匹配）
            pic_urls = re.findall(r'murl&quot;:&quot;(.*?)&quot;', response.text)
            if not pic_urls:
                pic_urls = re.findall(r'"contenturl":"(.*?)"', response.text)
            if not pic_urls:
                pic_urls = re.findall(r'"imgurl":"(.*?)"', response.text)
            
            if not pic_urls:
                print(f"第 {offset//35 + 1} 页没有找到图片")
                break
            
            print(f"第 {offset//35 + 1} 页找到 {len(pic_urls)} 个图片链接")
            
            for img_url in pic_urls:
                if downloaded >= num_images:
                    break
                
                img_url = img_url.replace('\\', '').replace('u0026', '&')
                
                # 过滤掉太小的图片
                if 'favicon' in img_url or 'logo' in img_url:
                    continue
                
                try:
                    # 随机延迟，避免被封
                    time.sleep(random.uniform(0.3, 0.8))
                    
                    img_data = requests.get(img_url, headers=headers, timeout=10)
                    if img_data.status_code == 200:
                        # 判断文件类型
                        content_type = img_data.headers.get('content-type', '')
                        if 'image' in content_type:
                            if 'jpeg' in content_type or 'jpg' in content_type:
                                ext = 'jpg'
                            elif 'png' in content_type:
                                ext = 'png'
                            else:
                                ext = 'jpg'
                        else:
                            ext = 'jpg'
                        
                        filename = os.path.join(save_dir, f'{keyword}_{downloaded+1}.{ext}')
                        with open(filename, 'wb') as f:
                            f.write(img_data.content)
                        downloaded += 1
                        print(f'已下载 {downloaded}/{num_images}')
                        
                except Exception as e:
                    continue
            
            offset += 35
            time.sleep(1)  # 翻页间隔
            
        except Exception as e:
            print(f"请求失败: {e}")
            break
    
    print(f"下载完成！成功下载 {downloaded} 张图片到 {save_dir}")
    return downloaded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--keyword', type=str, required=True, help='搜索关键词')
    parser.add_argument('--num', type=int, default=30, help='下载数量')
    parser.add_argument('--save_dir', type=str, default='./dataset', help='保存目录')
    args = parser.parse_args()
    
    save_path = os.path.join(args.save_dir, args.keyword)
    download_bing_images(args.keyword, save_path, args.num)


if __name__ == '__main__':
    main()
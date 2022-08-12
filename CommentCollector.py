from googleapiclient.discovery import build
import APIKeys

youtube = build('youtube', 'v3', developerKey=APIKeys.API_KEYS[1])


def get_comments(video_id, max_page):
    counter = 1
    video_response = youtube.commentThreads().list(
        part='snippet,replies',
        videoId=video_id,
        maxResults=100
    ).execute()

    comments = []
    # iterate video response
    while video_response:
        for item in video_response['items']:
            # Extracting comments
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        if counter < max_page:
            if 'nextPageToken' in video_response:
                video_response = youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id
                ).execute()
                counter += 1
            else:
                break
        else:
            break
    return comments


if __name__ == "__main__":
    # importing the modules
    import logging
    logging.basicConfig(filename='comments.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    from tqdm import tqdm

    from pathlib import Path    
    import pandas as pd

    # setting the paths
    file_path = "file_path"
    error_file_path = 'error_file_path'


    # reading the data
    df = pd.read_csv(file_path, sep="\t")

    # getting the comments
    comment_id = {"comment": [], "video_id": []}
    req_c = 0
    failed = 0
    for video_id in tqdm(df['id.videoId'].to_list(),leave=True):
        req_c += 1 
        try:
            comments = get_comments(video_id, 5)  # 5 requests per video id, 100 comments per request
            for comment in comments:
                comment_id['comment'].append(comment)
                comment_id['video_id'].append(video_id)
        except Exception as e:
            print(video_id, e)
            logging.error(msg=str(video_id) + " : " + str(e)+'\n')
            failed += 1
            with open(error_file_path,'a')as ef:
                ef.write(f"{video_id}\n")

    print(file_path)
    print("Number of requests:",req_c)
    print("Number of failed requests:",failed)       

    id_comment_df = pd.DataFrame(comment_id)
    output_file_path = f'\\{Path(file_path).name[:-4]}_comments.tsv'
    id_comment_df.to_csv(output_file_path, sep='\t')








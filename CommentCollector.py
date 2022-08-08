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
    vi_id = "wk63ZY_GVzA"
    print(get_comments(vi_id, 5))





import xml.etree.ElementTree as ET
from datetime import datetime

from src.database import DatabaseManager
from src.entity.article import Article, Content
from src.services.article_service import ArticleService


def parse_articles_from_xml(file_path: str) -> list[Article]:
    """Parse a WordPress export XML file and return a list of Article objects.

    This function is designed to handle XML files exported from services like
    note.com, which use a WordPress-compatible RSS feed format.

    The XML structure is expected to be as follows:
    <rss>
        <channel>
            <title>Blog Title</title>
            <link>Blog URL</link>
            ...
            <item>
                <title>Article Title</title>
                <link>Article Link</link>
                <pubDate>Mon, 23 Mar 2020 08:00:00 +0000</pubDate>
                <dc:creator>Author Name</dc:creator>
                <content:encoded><![CDATA[<p>Article content...</p>]]></content:encoded>
                <wp:post_id>123</wp:post_id>
                <wp:post_date>2020-03-23 08:00:00</wp:post_date>
                <wp:status>publish</wp:status>
                ...
            </item>
            ...
        </channel>
    </rss>

    The function iterates through each <item> tag within the <channel>
    and extracts its data into an Article object. It handles different
    date formats for <pubDate> and <wp:post_date>.

    Parameters
    ----------
    file_path : str
        The path to the XML file.

    Returns
    -------
    list[Article]
        A list of Article objects parsed from the file.
        Returns an empty list if the file is not found or a parse error occurs.
    """
    articles: list[Article] = []

    # XMLの名前空間を辞書として定義します
    namespaces = {
        "wp": "http://wordpress.org/export/1.2/",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    try:
        # XMLファイルを解析します
        tree = ET.parse(file_path)
        root = tree.getroot()

        # 各記事(<item>)をループ処理します
        for item in root.findall("channel/item"):
            # pubDateとpost_dateで日付フォーマットが異なるため、別々に処理します
            pub_date_str = item.find("pubDate").text
            post_date_str = item.find("wp:post_date", namespaces).text

            pub_date = datetime.now()
            if pub_date_str:
                # 例: 'Mon, 23 Mar 2020 08:00:00 +0000'
                pub_date = datetime.strptime(pub_date_str, "%a, %d %b %Y %H:%M:%S %z")

            post_date = datetime.now()
            if post_date_str:
                # 例: '2016-01-25 00:05:07'
                post_date = datetime.strptime(post_date_str, "%Y-%m-%d %H:%M:%S")

            content_html = item.find("content:encoded", namespaces).text or ""
            content_obj = Content(html=content_html)

            # findメソッドと名前空間を使って各要素からテキストを取得します
            article = Article(
                title=item.find("title").text or "",
                link=item.find("link").text or "",
                creator=item.find("dc:creator", namespaces).text or "",
                content=content_obj,  # Contentオブジェクトを渡す
                post_id=int(item.find("wp:post_id", namespaces).text or 0),
                pub_date=pub_date,
                post_date=post_date,
                post_type=item.find("wp:post_type", namespaces).text or "",
                status=item.find("wp:status", namespaces).text or "",
            )
            articles.append(article)

    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりませんでした。")
    except ET.ParseError:
        print(f"エラー: ファイル '{file_path}' のXML解析に失敗しました。")

    return articles


def parse_args():
    """Parse command line arguments for the script.

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments.
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Parse WordPress export XML files.")
    parser.add_argument(
        "--xml",
        type=Path,
        required=True,
        help="Path to the XML file to parse.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    xml_file = args.xml

    # 記事リストを取得
    article_list = parse_articles_from_xml(xml_file)
    print(f"Found {len(article_list)} articles.")

    # データベースに保存
    db_manager = DatabaseManager()
    article_service = ArticleService(db_manager)
    article_service.save_articles(article_list)
    print(f"Saved {len(article_list)} articles to database.")
    print(f"Total articles in database: {article_service.get_article_count()}")

use std::{fs::File, io::{Cursor, Read}, net::TcpStream, path::Path};
use url::Url;



pub async fn read_from_url(url: &Url) -> anyhow::Result<Box<dyn Read>>  {
    match url.scheme() {
        #[cfg(feature = "http")]
        "http"| "https" => read_http(url).await,
        #[cfg(feature = "ssh")]
        "ssh" => read_ssh(url).await,
        "file" => read_file(url.path()).await,
        _ => anyhow::bail!("Unsupported scheme: {}", url.scheme())
    }
}

#[cfg(feature = "ssh")]
async fn read_ssh(url:&Url) -> anyhow::Result<Box<dyn Read>> {
    use ssh2::Session;
    let port = url.port().unwrap_or(22);
    let tcp = TcpStream::connect((url.domain().unwrap(),port))?;
    let mut sess = Session::new()?;
    sess.set_tcp_stream(tcp);
    sess.handshake()?;
    sess.userauth_agent(url.username())?;

    let ( mut remote_file, _stat) = sess.scp_recv(Path::new(url.path()))?;

    let mut data = Vec::with_capacity(_stat.size() as usize);
    remote_file.read_to_end(&mut data)?;

    remote_file.send_eof()?;
    remote_file.wait_eof()?;
    remote_file.close()?;
    remote_file.wait_close()?;

    return Ok(Box::new(Cursor::new(data)));
}

#[cfg(feature = "http")]
async fn read_http(url:&Url) ->  anyhow::Result<Box<dyn Read>>  {
    let request = ehttp::Request::get(url);
    let resp = ehttp::fetch_async(request).await.map_err(|e| anyhow::anyhow!(e))?;
    return Ok(Box::new(Cursor::new(resp.bytes)));
}

async fn read_file(path:&str) ->  anyhow::Result<Box<dyn Read>>  {
    let path = Path::new(path);
    let file = File::open(path)?;
    return Ok(Box::new(file));
}
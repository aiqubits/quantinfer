//! Minimal HTTP control plane for the optional online runtime.

use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

fn main() {
    let bind = env::args()
        .skip(1)
        .collect::<Vec<_>>()
        .windows(2)
        .find(|pair| pair[0] == "--bind")
        .map(|pair| pair[1].clone())
        .unwrap_or_else(|| "127.0.0.1:8080".to_owned());
    let listener = TcpListener::bind(&bind).unwrap_or_else(|error| {
        eprintln!("quantinfer-server: cannot bind {bind}: {error}");
        std::process::exit(1);
    });
    println!("quantinfer-server listening on {bind}");
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let _ = handle_connection(stream);
            }
            Err(error) => eprintln!("quantinfer-server: connection error: {error}"),
        }
    }
}

fn handle_connection(mut stream: TcpStream) -> std::io::Result<()> {
    let mut request = [0_u8; 8192];
    let length = stream.read(&mut request)?;
    let response = response_for_request(std::str::from_utf8(&request[..length]).unwrap_or(""));
    stream.write_all(response.as_bytes())
}

fn response_for_request(request: &str) -> String {
    let path = request.split_whitespace().nth(1).unwrap_or("/");
    let (status, content_type, body) = match path {
        "/healthz" => ("200 OK", "application/json", r#"{"status":"ok"}"#),
        "/v1/models" => ("200 OK", "application/json", r#"{"data":[]}"#),
        "/metrics" => (
            "200 OK",
            "text/plain; version=0.0.4",
            "quantinfer_requests_total 0\n",
        ),
        _ => (
            "404 Not Found",
            "application/json",
            r#"{"error":"not_found"}"#,
        ),
    };
    format!("HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len())
}

#[cfg(test)]
mod tests {
    use super::response_for_request;

    #[test]
    fn serves_health_models_metrics_and_not_found_routes() {
        assert!(response_for_request("GET /healthz HTTP/1.1\r\n\r\n").contains("200 OK"));
        assert!(response_for_request("GET /v1/models HTTP/1.1\r\n\r\n").contains("{\"data\":[]}"));
        assert!(response_for_request("GET /metrics HTTP/1.1\r\n\r\n")
            .contains("quantinfer_requests_total"));
        assert!(response_for_request("GET /unknown HTTP/1.1\r\n\r\n").contains("404 Not Found"));
    }
}

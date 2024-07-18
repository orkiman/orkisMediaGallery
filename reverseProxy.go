// reverseProxy.go
package main

import (
	"net/http"
	"net/http/httputil"
	"net/url"
)

// NewReverseProxy creates a new reverse proxy handler
func NewReverseProxy(targetHost string) http.Handler {
	targetURL, _ := url.Parse(targetHost)
	return httputil.NewSingleHostReverseProxy(targetURL)
}

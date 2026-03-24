var pc = null;
var offerRequestId = 0;
var offerAbortController = null;

function cancelPendingOffer() {
    offerRequestId += 1;
    if (offerAbortController) {
        offerAbortController.abort();
        offerAbortController = null;
    }
}

function syncSilenceGateState() {
    return Promise.resolve();
}

function emitLiveTalkingEvent(name, detail) {
    window.dispatchEvent(new CustomEvent(name, { detail: detail || {} }));
}

function logState(prefix, value) {
    console.log(prefix + ':', value);
}

function setSessionId(value) {
    var sessionInput = document.getElementById('sessionid');
    if (sessionInput) {
        sessionInput.value = value;
    }
    emitLiveTalkingEvent('livetalking:sessionchange', { sessionid: value });
}

function closePeerConnection() {
    if (!pc) {
        return;
    }

    var currentPc = pc;
    pc = null;
    setTimeout(function () {
        currentPc.close();
    }, 200);
}

function setWebRTCControls(startVisible, stopVisible) {
    var startButton = document.getElementById('start');
    var stopButton = document.getElementById('stop');

    if (startButton) {
        startButton.style.display = startVisible ? 'inline-block' : 'none';
    }
    if (stopButton) {
        stopButton.style.display = stopVisible ? 'inline-block' : 'none';
    }
}

function unlockMediaPlayback() {
    var audio = document.getElementById('audio');
    var video = document.getElementById('video');

    if (video && video.srcObject) {
        video.play().catch(function () {});
    }
    if (audio && audio.srcObject) {
        audio.play().catch(function () {});
    }
}

document.addEventListener('pointerdown', unlockMediaPlayback, true);
document.addEventListener('keydown', unlockMediaPlayback, true);

function negotiate() {
    cancelPendingOffer();
    var requestId = offerRequestId;
    var controller = typeof AbortController !== 'undefined' ? new AbortController() : null;
    offerAbortController = controller;
    pc.addTransceiver('video', { direction: 'recvonly' });
    pc.addTransceiver('audio', { direction: 'recvonly' });
    emitLiveTalkingEvent('livetalking:connectionstate', { state: 'negotiating' });
    return pc.createOffer().then(function (offer) {
        return pc.setLocalDescription(offer);
    }).then(function () {
        return new Promise(function (resolve) {
            if (pc.iceGatheringState === 'complete') {
                resolve();
            } else {
                var checkState = function () {
                    if (pc.iceGatheringState === 'complete') {
                        pc.removeEventListener('icegatheringstatechange', checkState);
                        resolve();
                    }
                };
                pc.addEventListener('icegatheringstatechange', checkState);
            }
        });
        }).then(function () {
            var offer = pc.localDescription;
            return fetch('/offer', {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type
                }),
                headers: {
                    'Content-Type': 'application/json'
                },
                method: 'POST',
                signal: controller ? controller.signal : undefined
            });
        }).then(function (response) {
            if (!response.ok) {
                return response.json().then(function (result) {
                    throw new Error(result.msg || ('Offer failed: ' + response.status));
            });
        }
        return response.json();
        }).then(function (answer) {
            if (requestId !== offerRequestId) {
                return;
            }
            setSessionId(answer.sessionid || 0);
            emitLiveTalkingEvent('livetalking:connectionstate', {
                state: 'ready',
                sessionid: answer.sessionid || 0
            });
            syncSilenceGateState().catch(function () {});
            return pc.setRemoteDescription(answer);
        }).catch(function (e) {
            if (requestId !== offerRequestId || (e && e.name === 'AbortError')) {
                return;
            }
            setSessionId(0);
            emitLiveTalkingEvent('livetalking:connectionstate', {
                state: 'error',
                error: String(e)
            });
        alert(e);
    });
}

function start() {
    cancelPendingOffer();
    closePeerConnection();
    setSessionId(0);

    var config = {
        sdpSemantics: 'unified-plan'
    };

    if (document.getElementById('use-stun').checked) {
        config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }];
    }

    pc = new RTCPeerConnection(config);
    emitLiveTalkingEvent('livetalking:connectionstate', { state: 'starting' });
    logState('WebRTC start config', config);

    pc.addEventListener('track', function (evt) {
        if (evt.track.kind === 'video') {
            var video = document.getElementById('video');
            video.srcObject = evt.streams[0];
            video.play().catch(function () {});
        } else {
            var audio = document.getElementById('audio');
            audio.srcObject = evt.streams[0];
            audio.play().catch(function () {});
        }
    });

    pc.addEventListener('connectionstatechange', function () {
        logState('pc.connectionState', pc.connectionState);
        emitLiveTalkingEvent('livetalking:connectionstate', { state: pc.connectionState });
        if (pc.connectionState === 'failed' || pc.connectionState === 'closed' || pc.connectionState === 'disconnected') {
            setSessionId(0);
        }
    });
    pc.addEventListener('iceconnectionstatechange', function () {
        logState('pc.iceConnectionState', pc.iceConnectionState);
        emitLiveTalkingEvent('livetalking:connectionstate', {
            state: 'ice-' + pc.iceConnectionState,
            iceConnectionState: pc.iceConnectionState
        });
    });
    pc.addEventListener('icegatheringstatechange', function () {
        logState('pc.iceGatheringState', pc.iceGatheringState);
    });

    setWebRTCControls(false, true);
    negotiate();
}

function stop() {
    cancelPendingOffer();
    setWebRTCControls(true, false);
    setSessionId(0);
    emitLiveTalkingEvent('livetalking:connectionstate', { state: 'stopped' });
    closePeerConnection();
}

window.addEventListener('pagehide', function () {
    cancelPendingOffer();
    closePeerConnection();
});

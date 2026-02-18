// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts-upgradeable/token/ERC721/ERC721Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/AccessControlUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/PausableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/security/ReentrancyGuardUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

/**
 * @title ProofOfAuthorship
 * @dev Soulbound Token contract for academic authorship verification
 * Features:
 * - Non-transferable tokens (Soulbound)
 * - Gas-optimized batch operations
 * - Upgradeable contract pattern
 * - Comprehensive event logging
 * - Role-based access control
 */
contract ProofOfAuthorship is 
    Initializable,
    ERC721Upgradeable, 
    AccessControlUpgradeable, 
    PausableUpgradeable, 
    ReentrancyGuardUpgradeable, 
    UUPSUpgradeable 
{
    using Counters for Counters.Counter;

    // Roles
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");
    bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");
    bytes32 public constant PAUSER_ROLE = keccak256("PAUSER_ROLE");

    // Token counter
    Counters.Counter private _tokenIdCounter;

    // Submission data structure
    struct Submission {
        bytes32 contentHash;        // SHA-256 hash of essay content
        bytes32 stylometricHash;    // Hash of stylometric signature
        string ipfsHash;            // IPFS hash for full content
        address student;            // Student wallet address
        uint256 timestamp;          // Submission timestamp
        uint8 authorshipScore;      // Authorship score (0-100)
        uint8 aiProbability;        // AI detection probability (0-100)
        bool verified;              // Verification status
        string institutionId;       // Institution identifier
        string courseId;            // Course identifier
    }

    // Mappings
    mapping(uint256 => Submission) public submissions;
    mapping(address => uint256[]) public studentSubmissions;
    mapping(bytes32 => uint256) public contentHashToTokenId;
    mapping(string => uint256[]) public institutionSubmissions;

    // Events
    event SubmissionMinted(
        uint256 indexed tokenId,
        address indexed student,
        bytes32 indexed contentHash,
        string ipfsHash,
        uint8 authorshipScore,
        uint8 aiProbability
    );

    event SubmissionVerified(
        uint256 indexed tokenId,
        address indexed verifier,
        bool verified
    );

    event BatchSubmissionsMinted(
        uint256[] tokenIds,
        address[] students,
        uint256 timestamp
    );

    event SubmissionUpdated(
        uint256 indexed tokenId,
        string newIpfsHash,
        uint8 newAuthorshipScore
    );

    // Custom errors for gas optimization
    error TokenNotTransferable();
    error UnauthorizedAccess();
    error InvalidSubmissionData();
    error DuplicateSubmission();
    error TokenNotFound();

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /**
     * @dev Initialize the contract (for upgradeable pattern)
     */
    function initialize(address admin) public initializer {
        __ERC721_init("ProofOfAuthorship", "POA");
        __AccessControl_init();
        __Pausable_init();
        __ReentrancyGuard_init();
        __UUPSUpgradeable_init();

        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(VERIFIER_ROLE, admin);
        _grantRole(UPGRADER_ROLE, admin);
        _grantRole(PAUSER_ROLE, admin);
    }

    /**
     * @dev Mint a new proof-of-authorship token
     */
    function mintProofToken(
        bytes32 _contentHash,
        bytes32 _stylometricHash,
        string memory _ipfsHash,
        address _student,
        uint8 _authorshipScore,
        uint8 _aiProbability,
        string memory _institutionId,
        string memory _courseId
    ) external onlyRole(VERIFIER_ROLE) whenNotPaused nonReentrant returns (uint256) {
        // Validate input data
        if (_contentHash == bytes32(0) || _student == address(0)) {
            revert InvalidSubmissionData();
        }

        // Check for duplicate submissions
        if (contentHashToTokenId[_contentHash] != 0) {
            revert DuplicateSubmission();
        }

        // Increment token counter
        _tokenIdCounter.increment();
        uint256 tokenId = _tokenIdCounter.current();

        // Create submission record
        submissions[tokenId] = Submission({
            contentHash: _contentHash,
            stylometricHash: _stylometricHash,
            ipfsHash: _ipfsHash,
            student: _student,
            timestamp: block.timestamp,
            authorshipScore: _authorshipScore,
            aiProbability: _aiProbability,
            verified: true,
            institutionId: _institutionId,
            courseId: _courseId
        });

        // Update mappings
        contentHashToTokenId[_contentHash] = tokenId;
        studentSubmissions[_student].push(tokenId);
        institutionSubmissions[_institutionId].push(tokenId);

        // Mint the token
        _safeMint(_student, tokenId);

        emit SubmissionMinted(
            tokenId,
            _student,
            _contentHash,
            _ipfsHash,
            _authorshipScore,
            _aiProbability
        );

        return tokenId;
    }

    /**
     * @dev Batch mint multiple tokens for gas optimization
     */
    function batchMintProofTokens(
        bytes32[] memory _contentHashes,
        bytes32[] memory _stylometricHashes,
        string[] memory _ipfsHashes,
        address[] memory _students,
        uint8[] memory _authorshipScores,
        uint8[] memory _aiProbabilities,
        string[] memory _institutionIds,
        string[] memory _courseIds
    ) external onlyRole(VERIFIER_ROLE) whenNotPaused nonReentrant returns (uint256[] memory) {
        uint256 length = _contentHashes.length;
        
        // Validate array lengths
        require(
            length == _stylometricHashes.length &&
            length == _ipfsHashes.length &&
            length == _students.length &&
            length == _authorshipScores.length &&
            length == _aiProbabilities.length &&
            length == _institutionIds.length &&
            length == _courseIds.length,
            "Array length mismatch"
        );

        uint256[] memory tokenIds = new uint256[](length);

        for (uint256 i = 0; i < length; i++) {
            // Validate input data
            if (_contentHashes[i] == bytes32(0) || _students[i] == address(0)) {
                revert InvalidSubmissionData();
            }

            // Check for duplicate submissions
            if (contentHashToTokenId[_contentHashes[i]] != 0) {
                revert DuplicateSubmission();
            }

            // Increment token counter
            _tokenIdCounter.increment();
            uint256 tokenId = _tokenIdCounter.current();
            tokenIds[i] = tokenId;

            // Create submission record
            submissions[tokenId] = Submission({
                contentHash: _contentHashes[i],
                stylometricHash: _stylometricHashes[i],
                ipfsHash: _ipfsHashes[i],
                student: _students[i],
                timestamp: block.timestamp,
                authorshipScore: _authorshipScores[i],
                aiProbability: _aiProbabilities[i],
                verified: true,
                institutionId: _institutionIds[i],
                courseId: _courseIds[i]
            });

            // Update mappings
            contentHashToTokenId[_contentHashes[i]] = tokenId;
            studentSubmissions[_students[i]].push(tokenId);
            institutionSubmissions[_institutionIds[i]].push(tokenId);

            // Mint the token
            _safeMint(_students[i], tokenId);

            emit SubmissionMinted(
                tokenId,
                _students[i],
                _contentHashes[i],
                _ipfsHashes[i],
                _authorshipScores[i],
                _aiProbabilities[i]
            );
        }

        emit BatchSubmissionsMinted(tokenIds, _students, block.timestamp);
        return tokenIds;
    }

    /**
     * @dev Update submission metadata (IPFS hash, scores)
     */
    function updateSubmission(
        uint256 _tokenId,
        string memory _newIpfsHash,
        uint8 _newAuthorshipScore
    ) external onlyRole(VERIFIER_ROLE) {
        if (!_exists(_tokenId)) {
            revert TokenNotFound();
        }

        Submission storage submission = submissions[_tokenId];
        submission.ipfsHash = _newIpfsHash;
        submission.authorshipScore = _newAuthorshipScore;

        emit SubmissionUpdated(_tokenId, _newIpfsHash, _newAuthorshipScore);
    }

    /**
     * @dev Verify or unverify a submission
     */
    function setVerificationStatus(
        uint256 _tokenId,
        bool _verified
    ) external onlyRole(VERIFIER_ROLE) {
        if (!_exists(_tokenId)) {
            revert TokenNotFound();
        }

        submissions[_tokenId].verified = _verified;
        emit SubmissionVerified(_tokenId, msg.sender, _verified);
    }

    /**
     * @dev Get submission details
     */
    function getSubmission(uint256 _tokenId) external view returns (Submission memory) {
        if (!_exists(_tokenId)) {
            revert TokenNotFound();
        }
        return submissions[_tokenId];
    }

    /**
     * @dev Get all submissions for a student
     */
    function getStudentSubmissions(address _student) external view returns (uint256[] memory) {
        return studentSubmissions[_student];
    }

    /**
     * @dev Get all submissions for an institution
     */
    function getInstitutionSubmissions(string memory _institutionId) external view returns (uint256[] memory) {
        return institutionSubmissions[_institutionId];
    }

    /**
     * @dev Get token by content hash
     */
    function getTokenByContentHash(bytes32 _contentHash) external view returns (uint256) {
        return contentHashToTokenId[_contentHash];
    }

    /**
     * @dev Get total number of tokens minted
     */
    function totalSupply() external view returns (uint256) {
        return _tokenIdCounter.current();
    }

    /**
     * @dev Override transfer functions to make tokens soulbound (non-transferable)
     */
    function transferFrom(address, address, uint256) public pure override {
        revert TokenNotTransferable();
    }

    function safeTransferFrom(address, address, uint256) public pure override {
        revert TokenNotTransferable();
    }

    function safeTransferFrom(address, address, uint256, bytes memory) public pure override {
        revert TokenNotTransferable();
    }

    function approve(address, uint256) public pure override {
        revert TokenNotTransferable();
    }

    function setApprovalForAll(address, bool) public pure override {
        revert TokenNotTransferable();
    }

    /**
     * @dev Pause contract functionality
     */
    function pause() external onlyRole(PAUSER_ROLE) {
        _pause();
    }

    /**
     * @dev Unpause contract functionality
     */
    function unpause() external onlyRole(PAUSER_ROLE) {
        _unpause();
    }

    /**
     * @dev Required for upgradeable contracts
     */
    function _authorizeUpgrade(address newImplementation) internal override onlyRole(UPGRADER_ROLE) {}

    /**
     * @dev Override required by Solidity for multiple inheritance
     */
    function supportsInterface(bytes4 interfaceId) public view override(ERC721Upgradeable, AccessControlUpgradeable) returns (bool) {
        return super.supportsInterface(interfaceId);
    }

    /**
     * @dev Token URI for metadata
     */
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        if (!_exists(tokenId)) {
            revert TokenNotFound();
        }

        Submission memory submission = submissions[tokenId];
        return string(abi.encodePacked("https://ipfs.io/ipfs/", submission.ipfsHash));
    }
}